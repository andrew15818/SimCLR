import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import argparse
import os
import wandb

from data import ImbalanceCIFAR10_index, ImbalanceCIFAR100_index, get_transforms
from models.model import SimCLR
from models.sdclr import SDCLR
from criterion import nt_xent
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str, default='latest', 
                   help='Name of the experiment')
parser.add_argument('--data_dir', type=str, default='./', 
                   help='Path to the dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                   default='cifar10', help='dataset to run')
parser.add_argument('--imb_ratio', type=int, default=100,
                  help='imbalance ratio between head/tail class.')
parser.add_argument('--split', type=int, choices=[1, 2, 3],
                   help='number fo the split to train on.')
parser.add_argument('--model', type=str, default='resnet18', 
                   help='Architecture of the backbone encoder')
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--epochs', type=int, default=1200, 
                   help='Training epochs.')
parser.add_argument('--save_freq', type=int, default=300,
                   help='How often (in epochs) to save the model.')
parser.add_argument('--resume', type=str, default='',
                   help='Resume training from a checkpoint')
parser.add_argument('--gpu', type=int, default=0,
                   help='GPU to use.')

parser.add_argument('--encoder_dim', type=int, default=512,
                   help='output of the backbone encoder')
parser.add_argument('--temperature', type=float, default=0.2,
                   help='Temperature in the loss function.')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=3e-4,
                   help='initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                   help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                   help='weight decay')
parser.add_argument('--bcl', action='store_true', default=False,
                   help='Train boosted contrastive learning')
parser.add_argument('--sdclr', action='store_true', default=False,
                   help='Train self-damaging contrastive learning.')
parser.add_argument('--ours', action='store_true', default=False,
                   help='Whether to use our model.')
parser.add_argument('--baseline',  action='store_true', default=False,
                   help='Test a baseline Simclr.')

def get_weights(z, **kwargs):
    # temporary
    return torch.ones_like(z).detach()

# Train one epoch
def train(
    model, dataloader, optimizer, contrastive, 
    criterion, scheduler, args, **kwargs):
    lossMeter = AverageMeter('loss')
    for i, ((x1, x2), targets) in enumerate(dataloader):
        x1 = x1.to(args.device)
        x2 = x2.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        
        z1 = model(x1)
        z2 = model(x2)
        
        logits, labels = contrastive(z1, z2)
        norms = torch.norm(z1 - z2, p=2, dim=1)
       
        if args.ours:
            w = get_weights(norms)
            loss = torch.mean(criterion(logits, labels) * 
                              torch.cat([w, w],dim=0))
        else:
            loss = torch.mean(criterion(logits, labels))
            
        lossMeter.update(loss.item())    
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        kwargs['normMeter'].add_batch_value(norms, targets)
       
    #wandb.log({"loss": loss.item(), "": })
    return lossMeter.avg

def main():
   
    args = parser.parse_args() 
    args.device = f'cuda:{args.gpu}' 
    
    train_transforms, test_transforms = get_transforms()
    # Load the dataset
    if args.dataset == 'cifar10':
        dataset = ImbalanceCIFAR10_index(root=args.data_dir, train=True, 
                                         transform=train_transforms, split_num=args.split,
                                        imb_ratio=args.imb_ratio)
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        dataset = ImbalanceCIFAR100_index(root=args.data_dir, train=True,
                                         transform=train_transforms, split_num=args.split, 
                                         imb_ratio=args.imb_ratio)
        args.num_classes = 100
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load the model
    if args.sdclr:
        model = SDCLR(num_class=args.encoder_dim, network=args.model)
    else:
        if args.model == 'resnet18':
            encoder = models.resnet18
        # Load our own model
        model = SimCLR(base_encoder=encoder, encoder_dim=args.encoder_dim, proj_hid_dim=128)
        
    model.to(args.device)
    # Optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    # Criterion
    contrastive = nt_xent(args.temperature)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize weights and biases
    method = 'sdclr' if args.sdclr else 'ours' if args.ours else 'baseline' if args.baseline else 'bcl'
    # Name of the directory to save
    save_dir = f'{args.experiment}/{method}_{args.dataset}_{args.model}_ratio{args.imb_ratio}_split{args.split}'
    print(f'Saving at {save_dir}')
    #wandb.init(
    #   project='MyReweighting' ,
    #    name=save_dir,
    #    config = {
    #        "learning_rate": args.lr,
    #        "optimizer": args.optimizer,
    #        "name": save_dir,
    #        "architecture":args.model,
    #        "dataset": args.dataset,
    #        "epochs": args.epochs,
    #        "batch_size": args.batch_size,
    #    }
    #)
    normMeter = ClassAverageMeter(args, dataset.get_sample_dist())
    losses = []
    model.train()
    for epoch in range(1, args.epochs):
        epoch_loss = train(model, train_loader, optimizer, 
              contrastive, criterion, 
              scheduler, args,
             normMeter=normMeter)
        print(f'Epoch {epoch}: Loss {epoch_loss:.4f}')
        losses.append(epoch_loss)
        normMeter.update()
        
        plot_category(normMeter.get_values(), epoch=epoch, 
                        ylabel='Norm difference',
                        filename='imgs/norm_difference.png')
        
        plot_by_class(normMeter.get_values(), epoch=epoch, 
                        ylabel='Norm difference by class',
                        filename='imgs/norm_difference_class.png')
        
        plot(losses, title='Losses')
        # Save the model
        state_dict = {
                'state_dict': model.state_dict(),
                'lr': scheduler.get_lr(),
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
                }
        save_checkpoint(state_dict, save_dir=os.path.join('runs', save_dir), filename='model.pth.tar')
        if epoch % args.save_freq == 0:
            save_checkpoint(state_dict, save_dir= os.path.join('runs', save_dir), filename=f'model_{epoch}.pth.tar')
            
def save_checkpoint(state_dict, save_dir, filename):
    if not os.path.exists(save_dir):
        print(f'Making directory {save_dir}')
        os.makedirs(save_dir)
    print(f'Saving file at {os.path.join(save_dir, filename)}')
    torch.save(state_dict, os.path.join(save_dir, filename))
    
if __name__ == '__main__':
    main()
