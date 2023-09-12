import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import argparse
import os
import wandb

from data.data import (ImbalanceCIFAR10_index, ImbalanceCIFAR100_index, 
                       BalancedCIFAR10, get_transforms, memoboosted_CIFAR10)
from models.model import SimCLR
from models.sdclr import SDCLR
from criterion import nt_xent
from utils import *
from data.lars import LARS

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
parser.add_argument('--lr', type=float, default=1e-3,
                   help='initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                   help='Momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                   help='weight decay')
parser.add_argument('--bcl', action='store_true', default=False,
                   help='Train boosted contrastive learning')
parser.add_argument('--sdclr', action='store_true', default=False,
                   help='Train self-damaging contrastive learning.')
parser.add_argument('--ours', action='store_true', default=False,
                   help='Whether to use our model.')
parser.add_argument('--baseline',  action='store_true', default=False,
                   help='Test a baseline Simclr.')
parser.add_argument('--balanced', action='store_true', default=False,
                   help='Use the original cifar dataset.')
parser.add_argument('--momentum_loss_beta', type=float, default=0.97)
parser.add_argument('--rand_k', type=int, default=1, help='k in randaugment')
parser.add_argument('--rand_strength', type=int, default=30, help='maximum strength')
parser.add_argument('--prune_percent', type=float, default=0, help='whole prune percentage')

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0
    if step <= warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps)/ (total_steps - warmup_steps) * np.pi))
    return lr

def get_weights(z, **kwargs):
    # temporary
    mean = z.mean()
    w = 1 - ((z - mean) / (z.max() - z.min()))
    return w

# Train one epoch
def train(
    model, dataloader, optimizer, contrastive, 
    criterion, scheduler, args, shadow, 
    momentum_loss, **kwargs):
    lossMeter = AverageMeter('loss')
    model.train()
    for i, ((x1, x2), idx, targets) in enumerate(dataloader):

        x1 = x1.to(args.device)
        x2 = x2.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        
        z1 = model(x1)
        z2 = model(x2)
        norms = torch.norm(z1 - z2, p=2, dim=1)
        logits, labels = contrastive(z1, z2)
        
       
        #w = get_weights(norms)
        loss = criterion(logits, labels) #* torch.cat([w, w],dim=0)
        #kwargs['weightMeter'].add_batch_value(norms, targets)
        for count in range(x1.shape[0]):
            if kwargs['epoch'] > 1:
                new_average = (1.0 - args.momentum_loss_beta * loss[count].clone().detach() + args.momentum_loss_beta * shadow[idx[count]])
            else:
                new_average = loss[count].clone().detach()
            shadow[idx[count]] = new_average
            momentum_loss[kwargs['epoch']-1, idx[count]] = new_average

        lossMeter.update(loss.mean().item())    
        loss.mean().backward()
        optimizer.step()
        scheduler.step()
        
        #kwargs['normMeter'].add_batch_value(norms, targets)
       
    #wandb.log({"loss": loss.item(), "": })
    return shadow, momentum_loss, lossMeter.avg

def main():
   
    args = parser.parse_args() 
    args.device = f'cuda:{args.gpu}' 
    
    train_transforms, test_transforms = get_transforms()
    # Load the dataset
    if args.dataset == 'cifar10':
        if args.balanced:
            dataset = BalancedCIFAR10(root=args.data_dir, train=True,
                                     transform=train_transforms)
            print(f'Length of the balanced dataset: {len(dataset)}')
        else:
            dataset = memoboosted_CIFAR10(args.data_dir, args, train=True, transform=train_transforms)
            args.num_classes = 10 
    train_loader = DataLoader(dataset, 
                    batch_size=args.batch_size,
                    shuffle=True
                )
    model = SimCLR(base_encoder='resnet18' , encoder_dim=args.num_classes, proj_hid_dim=128)
        
    model.to(args.device)
    optimizer = LARS(model.parameters(), lr=args.lr)
    # Optimizer and scheduler
    #if args.optimizer == 'adam':
    #    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #else:
    #    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1, 1e-6/args.lr, warmup_steps=10*len(train_loader)))
    # Criterion
    contrastive = nt_xent(args.temperature)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize weights and biases
    method = 'bcl'
    
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
    #normMeter = ClassAverageMeter(args)
    #weightMeter = ClassAverageMeter(args)
    shadow = torch.zeros(len(dataset)).to(args.device)
    momentum_loss = torch.zeros(args.epochs, len(dataset)).to(args.device)
    losses = []
    cum_time = 0
    model.train()
    for epoch in range(1, args.epochs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        shadow, momentum_loss, epoch_loss = train(model, train_loader, optimizer, 
              contrastive, criterion, 
              scheduler, args,
              shadow, 
              momentum_loss,
              epoch=epoch,
             #normMeter=normMeter,
             #weightMeter=weightMeter
             )
        end.record()
        dataset.update_momentum_weight(momentum_loss, epoch)
        torch.cuda.synchronize()
        latest = start.elapsed_time(end) / 1000
        cum_time += latest
        print(f'Epoch {epoch}: Loss {epoch_loss:.4f}, lr: {scheduler.get_lr()}, time: {latest:.4f}, avg_time{cum_time/epoch:.4f}')
        losses.append(epoch_loss)
        #normMeter.update()
        
        #plot_category(normMeter.get_values(), epoch=epoch, 
        #                ylabel='Norm difference',
        #                filename='imgs/norm_difference.png')
        #
        #plot_by_class(normMeter.get_values(), epoch=epoch, 
        #                ylabel='Norm difference by class',
        #                filename='imgs/norm_difference_class.png')
        
        plot(losses, title='Losses')
        if args.ours:
            #weightMeter.update()
            plot_category(weightMeter.get_values(), epoch=epoch,
                    ylabel='Weights per category',
                    title='Weights',
                    filename='imgs/weights.png')
            plot_by_class(weightMeter.get_values(), epoch=epoch,
                    title='Weights',
                    ylabel='Weights per class.',
                    filename='imgs/weights_class.png')
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
