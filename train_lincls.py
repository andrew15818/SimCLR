import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from model import SimCLR
from utils import *
from collections import defaultdict

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../../cvdl2022',
    help='Path to the dataset.')
parser.add_argument('--arch', type=str, default='resnet18', 
    help='backbone architecture')
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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()
    model  = SimCLR(models.__dict__[args.arch], args.out_dim)
    if args.checkpoint != '':
        checkpoint = torch.load('runs/checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
    model = model.encoder
    model.fc= nn.Linear(args.out_dim, args.num_classes)
    model.to(device)

    # Freeze the parameters except for last linear layer
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    # Train, test on balanced datasets
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Just one number since the datasets are balanced and contain equal number of samples/class
    trainClassCounts = len(train_loader.dataset) / args.num_classes
    testClassCounts = len(test_loader.dataset) / args.num_classes
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
        print(f'{sorted(trainClassAccs.items())}')
if __name__ == '__main__':
    main()