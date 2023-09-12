import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
import torch.optim as optim
import torchvision.models as models
import argparse
import wandb
import os
import sys

sys.path.append('../')
from data.data import INaturalistSSL, get_transforms
from models.model import SimCLR
from models.sdclr import SDCLR
from criterion import nt_xent
from utils import * 
from data.lars import LARS
from criterion import nt_xent

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str, default='inat_simclr',
                    help='Name of the experiment to run.')
parser.add_argument('--data_dir', type=str, default='../../datasets',
                    help='Path to the dataset roots.')
parser.add_argument('--model', type=str, default='resnet50',
                    help='backbone architecture')
parser.add_argument('--epochs', type=int, default=90, 
                    help='Training epochs for the dataset')
parser.add_argument('--lr', type=float, default=5.0,
                   help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                   help='momentum')
parser.add_argument('--weight_decay', type=float, default=.0005,
                   help='weight decay')
parser.add_argument('--encoder_dim', type=int, default=8142,
                   help='Backbone encoder output (default 8142)')
parser.add_argument('--projector_dim', type=int, default=128,
                   help='Output dim for the projector (default 128)')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--checkpoint', type=str, default='', 
                    help='Path from where to load the pre-trained model.')
parser.add_argument('--save_freq', type=int, default=30,
                    help='How often to save the model during training.')
parser.add_argument('--gpu', type=int, default=1,
                    help='Which gpu to use during training')
parser.add_argument('--temperature', type=int, default=0.2,
                    help='Temperature to use in contrastive loss')
parser.add_argument('--optimizer', type=str, default='lars',
                    help='Optimizer (default lars)')
parser.add_argument('--baseline', action='store_true', default=False, 
                    help='Train a base simclr')
parser.add_argument('--ours', action='store_true', default=False,
                   help='Train lotar model.')

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0
    if step <= warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps)/ (total_steps - warmup_steps)* np.pi))
    return lr 
# Get the weights for our model
def get_weights(z, **kwargs):
    mean = z.mean()
    w = 1 + ((z-mean) / (z.max() - z.min()))
    return w 

def main():
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}'
    train_transforms, test_transforms = get_transforms(crop_size=32)
    
    # Load dataset
    train_dataset = INaturalistSSL(root=args.data_dir,
                                   filename='./data/iNaturalist18_train.txt',
                                   transform=train_transforms
                                  )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 

    if args.baseline or args.ours: 
        model = SimCLR(base_encoder=args.model,
                      encoder_dim=args.encoder_dim,
                      proj_hid_dim=args.projector_dim).to(args.device)
    method = 'baseline' if args.baseline else 'lotar'
    
    # Optimizer
    optimizer = LARS(model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
                    lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 
                    1, 1e-6/args.lr, warmup_steps=10*len(train_loader)))
    start_epoch = 0
    if args.checkpoint != '':
        print(f'Loading {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        print('--> Loaded model parameters')
        start_epoch = checkpoint['epochs'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for i in range((start_epoch - 1) * len(train_loader)):
            scheduler.step()
        print('--> Loaded optimizer and scheduler.')
        
    # Loss
    contrastive = nt_xent(args.temperature)
    criterion = nn.CrossEntropyLoss(reduction='none') # Return the loss for each sample

    save_dir = f'{args.experiment}/{method}_inat18'

    # Train
    for epoch in range(start_epoch, args.epochs+1):
        epoch_loss = train(model, train_loader, optimizer, scheduler, contrastive, criterion, args, epoch)
        # ToDo: Save the model every epoch according to its destination

        state_dict = {
            'model': model.state_dict(),
            'lr': scheduler.get_lr(),
            'optimizer': optimizer.state_dict(),
            'epochs': epoch
        }
        save_checkpoint(state_dict, save_dir, 'model.pth')
       

def save_checkpoint(state_dict, save_dir, filename):
    if not os.path.exists(os.path.join('../runs',save_dir)):
        print(f'Making directory {save_dir}')
        os.makedirs(os.path.join('../runs/', save_dir))
    print(f"Saving file at {os.path.join('../runs/', save_dir, filename)}")
    torch.save(state_dict, os.path.join('../runs/', save_dir, filename))

def train(model, loader, optimizer, scheduler, contrastive, criterion, args, epoch):
    model.train()
    loss_meter = AverageMeter('loss')
    for batch, ((v1, v2), targets) in enumerate(loader):
        v1 = v1.to(args.device)
        v2 = v2.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()

        z1 = model(v1)
        z2 = model(v2)

        logits, labels = contrastive(z1, z2)
        # Multiply by our weights
        if args.ours:
            norms = torch.norm(z1 - z2, p=2, dim=1)
            w = get_weights(norms)
            loss = torch.mean(criterion(logits, labels) * 
                             torch.cat([w, w], dim=0))
        else:
            loss = torch.mean(criterion(logits, labels))

        loss.backward()
        loss_meter.update(loss.item())
        optimizer.step()
        scheduler.step()
        if batch % 100 == 0:
            print(f'\tEpoch {epoch} batch {batch}/{len(loader)} Avg loss: {loss_meter.avg:.4f} lr: {scheduler.get_lr()[0]:.4f}')
    return loss_meter.avg

if __name__ == '__main__':
    print('In here')
    main()
