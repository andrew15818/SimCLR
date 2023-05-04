import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import argparse
import data
from model import SimCLR
from criterion import nt_xent
from utils import *

model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__") and
        callable(models.__dict__[name]))

parser =  argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet18', help='encoder architecture.')
parser.add_argument('--lr', type=float, default=0.0003, help='initial lr')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--seed', type=int, default=0,
        help='Seed to split the classes along')
parser.add_argument('--data-dir', type=str, default='../../cvdl2022', help='path to the dataset')
parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset to use.')
parser.add_argument('--balanced', action='store_true', default=False, 
        help='Whether to use the balanced version of dataset.')
parser.add_argument('--encoder_dim', type=int, default=512,
    help='output of Resnet backbone.')
parser.add_argument('--proj_hid_dim', type=int, default=128,
    help='Projector hidden/output layer dim.')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--num-classes', type=int, default=10, help='Used in plotting functions')
parser.add_argument('--epochs', type=int, default=400, help='Epochs to train the model')
parser.add_argument('--resume', type=str, default='', help='Resume training from a checkpoint')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--temperature', type=float, default=0.3)


def get_weights(values, *args):
    mean = values.mean()
    w = 1 - ((values - mean)/ (values.max() - values.min()))
    return w.detach()

def train(model, loader, criterion, optimizer, scheduler, args, **kwargs):
    model.train()
    lossMeter = AverageMeter('loss')
    accMeter1 = AverageMeter('Top1')
    accMeter5 = AverageMeter('Top5')
    for i, ((x1, x2), targets) in enumerate(loader):
        x1 = x1.to(args.device)
        x2 = x2.to(args.device)
        
        optimizer.zero_grad()
        z1 = model(x1)
        z2 = model(x2)

        norms = torch.square(torch.norm((z1- z2), p=2, dim=1))
        sims = F.cosine_similarity(z1, z2, dim=1) 

        logits, labels = kwargs['info_nce'](z1, z2)
        w = get_weights(sims)
        loss = torch.mean(criterion(logits, labels) *
                torch.cat([w,w], dim=0))
        loss.backward()
        optimizer.step()
        scheduler.step()
        lossMeter.update(loss.item())
        
        # Get the measurements
        z1norm = F.normalize(z1, dim=1)
        z2norm = F.normalize(z2, dim=1)
        
        
        kwargs['normMeter'].add_batch_value(norms, targets)
        kwargs['simMeter'].add_batch_value(sims, targets)
        kwargs['weightMeter'].add_batch_value(w, targets)
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        accMeter1.update(top1.item(), targets.size(0))
        accMeter5.update(top5.item(), targets.size(0))


        
        if i % 20 == 0:
            print(f'Batch [{i+1}/{len(loader)}] Loss: {lossMeter.avg:.4f} Top1: {accMeter1.avg:.4f} Top5: {accMeter5.avg:.4f}')
    return lossMeter.avg, accMeter1.avg

def test():
    pass

def update_meters(*args):
    for meter in args:
        meter.update()

def main():
    args = parser.parse_args()
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    train_augs, test_augs = data.get_transforms()
    if args.dataset == 'cifar10':
        if args.balanced:
            print('Loading **Balanced** CIFAR10')
            dataset = data.BalancedCIFAR10
        else:
            dataset = data.ImbalanceCIFAR10
    elif args.dataset == 'cifar100':
            dataset = data.ImbalanceCIFAR100
    print(dataset)
    train_dataset = dataset(
            args.data_dir, 
            train=True, 
            download=True, 
            transform=train_augs,
            shuffleClasses=False)
    train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, shuffle=True)
    test_dataset= datasets.CIFAR10(
            args.data_dir,
            train=False,
            transform=test_augs)
    model = SimCLR(
        models.__dict__[args.arch],encoder_dim=args.encoder_dim, 
        proj_hid_dim=args.proj_hid_dim)
    
       
    # Replace the first conv layer only for cifar10
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        model.encoder.conv1 = nn.Conv2d(in_channels=3, out_channels=64, 
                                kernel_size=3, stride=1)
        model.avgpool = nn.Identity()
    
    model.to(args.device)
    info_nce = nt_xent(args.temperature)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    startEpoch = 0

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
                model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader))

    # Load pre-trained model
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        optimizer.load_state_dict(checkpoint['optimizer'])
        startEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded Model, will start from epoch {startEpoch}')
     
    trainLosses, trainAccs  = [], []
    normMeter = ClassAverageMeter(args, train_dataset.get_cls_num_dict())
    simMeter = ClassAverageMeter(args, train_dataset.get_cls_num_dict())
    weightMeter = ClassAverageMeter(args, train_dataset.get_cls_num_dict())

    for epoch in range(startEpoch, args.epochs):
        loss, acc = train(
            model, train_loader, 
            criterion, optimizer, 
            scheduler, args,
            normMeter=normMeter,
            simMeter=simMeter,
            weightMeter=weightMeter,
            info_nce=info_nce)
        print(f'Epoch {epoch} Average Loss: {loss:.4f} Top1: {acc:.4f}')
        trainLosses.append(loss)
        trainAccs.append(acc)

        # Save model 
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'arch': args.arch,
        }, filename='runs/checkpoint.pth.tar')
        # Update values we're tracking
        update_meters(normMeter, simMeter, weightMeter)
        plot(trainLosses)
        plot(trainAccs,title='{args.dataset} Training Accuracies', 
                ylabel='Accuracy', filename='imgs/accuracies.png')
        plot_category(normMeter.get_values(), ylabel=r'$d_i$', 
                title=f'{args.dataset} Norm difference', epoch=epoch, 
                filename='imgs/norm_difference.png')
        plot_category(simMeter.get_values(), 
                ylabel='Cosine Similarity', title=f'{args.dataset} Cosine Similarity between corresponding views.', 
                epoch=epoch,
                filename='imgs/cosine_sims.png')
        plot_category(weightMeter.get_values(), 
                ylabel='weight', title=f'{args.dataset} Weights per cateogry', 
                epoch=epoch, filename='imgs/weights.png')
    print(args)
if __name__ == '__main__':
    main()
