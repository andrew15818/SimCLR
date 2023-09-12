import torch
import torch.nn as nn
from models.resnet import resnet18,resnet50
#import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from models.model import SimCLR
from models.sdclr import SDCLR
from models.resnet_prune_multibn import prune_resnet18_dual
from data.data import ImbalanceCIFAR10_index, ImbalanceCIFAR100_index
from utils import * 

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../datasets')
parser.add_argument('--checkpoint', type=str, default='./model.pth.tar', 
                   help='Path to pretrained model')
parser.add_argument('--arch', type=str, default='resnet18',
                   help='backbone architecture')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10'])
parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3], 
                   help='Data split to load')
parser.add_argument('--ours', action='store_true', default=False,
                   help='Use our method')
parser.add_argument('--imb_ratio', type=int, default=100,
                   help='Imbalance ratio used during training')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='Baseline SimCLR')
parser.add_argument('--sdclr', action='store_true', default=False,
                   help='Self-damaging contrastive learning')
parser.add_argument('--bcl', action='store_true', default=False,
                   help='Boosted Contrastive Learning')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--epochs', type=int, default=90,
                   help='Epochs to fine-tune the classifier.')

def get_class_dist(args):
    targets = np.load(f'splits/{args.dataset}_split{args.split}_imbfac{args.imb_ratio}_targets.npy')
    class_dist = {}
    print(f'Loading split {args.split}')
    for i in range(args.num_classes):
        class_dist[i] = np.count_nonzero(targets == i)
        
    return class_dist


# Write the final accuracies to a file
def save_results(args, cat_accs, testClassAccs):
    save_dir = '/'.join(i for i in args.checkpoint.split('/')[:-1])
    filename = os.path.join(save_dir, 'results.txt')
    with open(filename, 'w') as f:
        f.write(f'Using split {args.split}\n')
        f.write(f'Category accs: {cat_accs}')
        f.write('\n')
        f.write(f'Per class Accs{testClassAccs}')
        f.write('\n')
    f.close()
    
def main():
   
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load the dataset -- balanced
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.ToTensor())
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.ToTensor())
        args.num_classes100
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    ## Load the model
    #model = resnet18(pretrained=False, num_classes=args.num_classes)
    #model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, 
    #                        kernel_size=3, stride=1)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    #print(f"Model trained for {checkpoint['epochs']} epochs")
    
    if args.baseline or args.ours or args.bcl:
        encoder_name = 'encoder'
        # Load the model
        model = resnet18(pretrained=False, num_classes=args.num_classes)
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, 
                            kernel_size=3, stride=1, bias=False)
        #model.normalize = nn.Identity()
    elif args.sdclr:
        encoder_name = 'backbone'
        model = prune_resnet18_dual(pretrained=False, num_classes=args.num_classes)
        
    for k in list(state_dict.keys()):
        if k.startswith(encoder_name) and not k.startswith(f'{encoder_name}.fc'):
            state_dict[k[len(f'{encoder_name}.'):]] = state_dict[k]
        del state_dict[k]
    
    log = model.load_state_dict(state_dict, strict=False)
    print(log)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2
    model.to(device)
    
    # Load the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Class distribution according to split
    class_dist = get_class_dist(args)
    latestClassAccs = None
    for epoch in range(args.epochs):
        lossMeter = AverageMeter('Loss')
        trainTop1 = AverageMeter('Train Top 1')
        testTop1 = AverageMeter('Test top 1')
        testTop5 = AverageMeter('Test top 5')
        trainClassAccs = defaultdict(int)
        testClassAccs = defaultdict(int)
        
        model.train()
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            lossMeter.update(loss.item())
            top1 = accuracy(logits, targets, topk=(1,))
            trainTop1.update(top1[0].item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            top = accuracy(logits, targets, topk=(1, 5))
            testTop1.update(top[0].item())
            testTop5.update(top[1].item())
            
            # Record per-class accuracies
            preds = torch.argmax(logits, dim=1)
            for j in range(targets.shape[0]):
                if preds[j].item() == targets[j].item():
                    testClassAccs[targets[j].item()] += 1
                    
        print(f"Epoch {epoch} Loss: {lossMeter.avg:.4f} Train Top1 : {trainTop1.avg:.4f}",
        f"Test Top1: {testTop1.avg:.4f} Test Top 5: {testTop5.avg:.4f}")
        cat_accs = print_category_accs(class_dist, testClassAccs, test_loader, args)
        latestClassAccs = testClassAccs
    print(latestClassAccs)
    save_results(args, cat_accs, latestClassAccs)
        
def print_category_accs(class_dist, test_accs, test_loader, args):
    imgs_per_class = len(test_loader.dataset) / args.num_classes
    if args.dataset == 'cifar10':
        cat_indices = {'Many': [0, 3],
                      'Medium': [3, 7],
                      'Few': [7, 10]}
    elif args.dataset == 'cifar100':
        cat_indices = {'Many': [0, 33],
                      'Medium': [33, 67],
                      'Few': [67, 100]}
    many_acc, med_acc, few_acc = 0, 0, 0
    for i, (classid, _) in enumerate(sorted(class_dist.items(), key=lambda p:p[1], reverse=True)):
        if i >= cat_indices['Many'][0] and i < cat_indices['Many'][1]:
            many_acc += (test_accs[classid] / imgs_per_class)
        elif i >= cat_indices['Medium'][0] and i < cat_indices['Medium'][1]:
            med_acc += (test_accs[classid] / imgs_per_class)
        elif i >= cat_indices['Few'][0] and i < cat_indices['Few'][1]:
            few_acc += (test_accs[classid] / imgs_per_class)
            
    many_acc /= cat_indices['Many'][1] - cat_indices['Many'][0]
    med_acc /= cat_indices['Medium'][1] - cat_indices['Medium'][0]
    few_acc /= cat_indices['Few'][1] - cat_indices['Few'][0]
    
    print(f'Many: {many_acc:.4f}\tMedium: {med_acc:.4f} Few: {few_acc:.4f}')
    return {'Many': many_acc, 'Medium': med_acc, 'Few': few_acc}
            
if __name__ == '__main__':
    main()
