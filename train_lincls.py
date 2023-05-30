import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from torch.utils.data import DataLoader
from model import SimCLR
from utils import *
from collections import defaultdict

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../datasets',
    help='Path to the dataset.')
parser.add_argument('--arch', type=str, default='resnet18', 
    help='backbone architecture')
parser.add_argument('--dataset_name', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=90, 
    help='Epochs to train a linear classifier on top of the representaitons.')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--checkpoint', type=str, default='',
    help='path to pretrained checkpoint')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--out_dim',type=int, default=512,
    help='Output of the encoder backbone.')
parser.add_argument('--num_classes', type=int, default=10)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#Open the file with the latest class distribution
def get_latest_class_distribution(args, path=None):
    targets = np.load(f'splits/{args.dataset_name}_imb.npy')
    class_dist = {}
    for i in range(args.num_classes):
        class_count = np.count_nonzero(targets == i)
        class_dist[i] = class_count
    print(class_dist)
    return class_dist

def main():
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    class_dist = get_latest_class_distribution(args) 
    
    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=args.num_classes)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=False, num_classes=args.num_classes)
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['state_dict']
    
    # Freeze the parameters except for last linear layer
    for k in list(state_dict.keys()):
        if k.startswith('encoder') and not k.startswith('encoder.fc'):
            state_dict[k[len('encoder.'):]] = state_dict[k]
        del state_dict[k]

    # Change the first conv layer to be able to load the state_dict
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=3, stride=1)
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    # Only train the last layer
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters())) 
    assert len(parameters) == 2
    model.to(device)
    print(model)
    # Train, test on balanced datasets
    if args.dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.ToTensor())
    elif args.dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Just one number since the datasets are balanced and contain equal number of samples/class
    trainClassCounts = len(train_loader.dataset) / args.num_classes
    testClassCounts = len(test_loader.dataset) / args.num_classes
    print(trainClassCounts, testClassCounts)
    cumClassAccs = defaultdict(list)
    for epoch in range(args.epochs):
        lossMeter = AverageMeter('Loss')
        trainTop1 = AverageMeter('Train Top 1')
        testTop1 = AverageMeter('Test Top 1')
        testTop5 = AverageMeter('Test top 5')
        trainClassAccs = defaultdict(int)
        testClassAccs = defaultdict(int)
        
        # Train
        model.train()
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, targets)
            top1 = accuracy(logits, targets, topk=(1,))
            trainTop1.update(top1[0].item())
            lossMeter.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure how many we got right
            preds = torch.argmax(logits, dim=1)
            for i in range(targets.shape[0]):
                if preds[i].item() == targets[i].item():
                    trainClassAccs[targets[i].item()] += 1
        
        # Test
        model.eval()
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)

            top1, top5 = accuracy(logits, targets, topk=(1,5))
            testTop1.update(top1.item())
            testTop5.update(top5.item())
            
            # Measure how many we got right
            preds = torch.argmax(logits, dim=1)
            for i in range(targets.shape[0]):
                if preds[i].item() == targets[i].item():
                    testClassAccs[targets[i].item()] += 1
        for (trid, trcor), (teid, tecor) in zip(trainClassAccs.items(), testClassAccs.items()):
            trainClassAccs[trid] = trcor / trainClassCounts
            testClassAccs[teid] = tecor / testClassCounts
        print(f'Epoch {epoch} Train loss: {lossMeter.avg:.4f}  Train Top1: {trainTop1.avg:.4f} \
        Test Top 1: {testTop1.avg:.4f} \
        Test Top5: {testTop5.avg:.4f}')
        # Print based on category
        print_category_accs(trainClassAccs, testClassAccs, class_dist, args)
        
    print(f'{sorted(testClassAccs.items(), key=lambda t: t[0])}')

# Order by Many, Medium, Few
def print_category_accs(trainAccs:dict, testAccs:dict, class_dist:dict, args):
    # Collect all the ordered accs into lists for easy index
    trainValues, testValues = [], []
    #print('Appending...')
    # At least one class has 0 acc, fill it in 
    if len(testAccs.keys()) != args.num_classes:
        for i in range(args.num_classes):
            if i not in testAccs:
                testAccs[i] = 0
        print(f'Fixed length to {len(list(testAccs.keys()))}')

    for classId, classCount in sorted(class_dist.items(), key=lambda t: t[1], reverse=True):
        #print(f'{classId}, ', end='')
        testValues.append(testAccs[classId])

    interval = args.num_classes // 3
    tv = np.array(testValues)
    #print(f'Many: 0-{interval} Medium: {interval}:{2*interval+1} Few: {2*interval+1}:{args.num_classes}')
    print(f'\tMany: {np.mean(tv[0:interval]):.4f}, Medium:{np.mean(tv[interval:2*interval+1]):.4f}, Few: {np.mean(tv[2*interval+1:]):.4f}')
if __name__ == '__main__':
    main()
