import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys
import argparse
import numpy as np
sys.path.append('../')

from models.resnet import resnet18, resnet50
from data.data import INaturalistSSL, get_transforms
from data.lars import LARS
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../../datasets/', 
                    help='Path to the dataset')
parser.add_argument('--checkpoint', type=str, 
                    default='../runs/simclr_inat/baseline_inat18/model.pth', 
                    help='Path to the pretrained checkpoint.')
parser.add_argument('--model', type=str, default='resnet50', 
                    help='Backbone architecture.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--optimizer', type=str, default='lars')
parser.add_argument('--lr', type=float, default=3e-4, 
                    help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=90, help='Epochs to train the linear classifier.')
parser.add_argument('--baseline', action='store_true', default=False, 
                    help='Whether testing the baseline.')
parser.add_argument('--classes', type=int, default=8192, 
                    help='Output number in classifier (class num)')
parser.add_argument('--ours', action='store_true', default=False, 
                    help='Whether using our method.')

# Measure "Many, Medium Few based on train set"
def measure_category_accs(dataset, test_accs):
    dist = dataset.get_data_dist()
    imgs_per_class = 3
    classes_per_cat = len(dataset) / 3
    acc_many, acc_med, acc_few = 0, 0, 0
    cat_indices = {'Many': [0, classes_per_cat],
                   'Medium':[classes_per_cat, 2* classes_per_cat],
                   'Few':[2*classes_per_cat, len(dataset)]
                  }
    for i, (classid, _) in enumerate(dist):
        if i < cat_indices['Many'][1]:
            acc_many += (test_accs[classid] / imgs_per_class)
        elif i >= cat_indices['Medium'][0] and i < cat_indices['Medium'][1]:
            acc_med += (test_accs[classid] / imgs_per_class)
        elif i >= cat_indices['Few'][0] and i < cat_indices['Few'][1]:
            acc_few += (test_accs[classid] / imgs_per_class)
        acc_many /= classes_per_cat
        acc_med /= classes_per_cat
        acc_few /= classes_per_cat
       
        print(f'Many: {acc_many}\t Medium: {acc_med}\t Few: {acc_few}')
        return {'Many': acc_many, 'Medium': acc_med, 'Few': acc_few}
    
def main():
    args = parser.parse_args()
    device = 'cuda:1'
    # Dataset, DataLoader
    _, test_transforms = get_transforms(crop_size=32)
    dataset = INaturalistSSL(root=args.data_dir, 
                             filename='./data/iNaturalist18_val.txt', 
                             transform=test_transforms,
                            test=True)
    # To measure the accuracy on the long-tailed classes in the training set.
    train_dataset = INaturalistSSL(root=args.data_dir,
                                  filename='./data/iNaturalist18_train.txt',
                                  transform=test_transforms,
                                  test=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load model
    if args.model == 'resnet50':
        model = resnet50(num_classes=args.classes)
    model.to(device)
    # Load all layers except final fc
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['model']
    print(f"Model has been trained for {checkpoint['epochs']} epochs.")
    
    for k in list(state_dict.keys()):
        if k.startswith('encoder') and not k.startswith(f'encoder.fc'):
            state_dict[k[len('encoder.'):]] = state_dict[k]
        del state_dict[k]
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2
    model.to(device)
    
    # Optimizer, scheduler, etc...
    #optimizer = #(model.parameters(), lr=args.lr, momentum=args.momentum)#
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
   
    # Evaluate
    for epoch in range(args.epochs):
        lossMeter = AverageMeter('loss')
        accMeter= AverageMeter('accuracy')
        testClassAccs = defaultdict(int)
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs[0].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            
            loss = criterion(out,labels)
            lossMeter.update(loss.item())
            top1 = accuracy(out, labels, topk=(1,))
            accMeter.update(top1[0].item())
            loss.backward()
            optimizer.step()
            
            # Measure per-class accuracy
            preds = torch.argmax(out, dim=1)
            for j in range(labels.shape[0]):
                if labels[j].item() == preds[j].item():
                    testClassAccs[labels[j].item()] += 1
                    
            # Print overall accuracy
            if i % 100 == 0:
                print(f'Epoch {epoch}: batch: {i}/{len(dataloader):.4f} loss: {loss.item():.4f} accuracy: {accMeter.avg:.4f}')
        cat_accs = measure_category_accs(train_dataset, testClassAccs) 
        save_results(args, cat_accs, accMeter.avg, testClassAccs)
        
def save_results(args, cat_accs, mean_acc, testClassAccs):
    save_dir = '/'.join(i for i in args.checkpoint.split('/')[:-1])
    filename = os.path.join(save_dir, 'results.txt')
    with open(filename, 'w') as f:
        f.write(f"Acc: {mean_acc:.4f}, Many: {cat_accs['Many']:.4f}, Medium: {cat_accs['Medium']:.4f}, Few: {cat_accs['Few']:.4f}\n")
        f.write(f'\n\n{testClassAccs}')
    f.close()
    
if __name__ == '__main__':
    main()
