from PIL import Image
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from .randaug import *

def _get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
        ])
    return color_distort
def get_transforms(crop_size=32):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        _get_color_distortion(s=0.5),
        transforms.ToTensor()])
    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor()
        ])
    return train_transforms, test_transforms

class ImbalanceCIFAR10_index(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, imb_type='exp', rand_number=0, train=True, transform=None, target_transform=None, split_num=1, imb_ratio=100, download=False):
        super(ImbalanceCIFAR10_index, self).__init__(root, train, transform, target_transform, download)
        self.root = root
        self.imb_type = imb_type
        self.imb_ratio = imb_ratio 
        self.train = train
        self.transform = transform
        self.split = split_num
        self.cls_num = 10
        

        self.load_data()
        
    def load_data(self):
        filename = f'splits/cifar{self.cls_num}_split{self.split}_imbfac{self.imb_ratio}'
        print(f'Loading {filename}')
        self.data = np.load(f'{filename}_data.npy')
        self.targets = np.load(f'{filename}_targets.npy')
    
    def get_sample_dist(self):
        unique, counts = np.unique(self.targets, return_counts=True)
        self.sample_dist = dict(zip(unique, counts))
        return self.sample_dist


    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        img = Image.fromarray(img)
        x1 = self.transform(img)
        x2 = self.transform(img)

        return [x1, x2], target
    
class ImbalanceCIFAR100_index(ImbalanceCIFAR10_index):
    pass

class BalancedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(BalancedCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

    def get_sample_dist(self):
        unique, counts = np.unique(self.targets, return_counts=True)
        self.sample_dist = dict(zip(unique, counts))
        return self.sample_dist


    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        x = Image.fromarray(x)
        x1 = self.transform(x)
        x2 = self.transform(x)
        
        return [x1, x2], y
        
class INaturalistSSL():
    def __init__(self, root, filename, transform=None, 
                 target_transform=None, download=False, test=False):
        #super(INaturalistSSL, self).__init__(root, version, target_type=target_type, 
                                             #transform=transform, target_transform=target_transform,
                                            #download=download)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = []
        self.labels = []
        self.test = test
        self.dist = None
        
        with open(filename) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
    
    def get_data_dist(self):
        if self.dist == None:
            self.dist = defaultdict(int)
            for label in self.labels:
                self.dist[label] += 1
        return sorted(self.dist.items(), key=lambda x: x[1], reverse=True)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        label = self.labels[idx]
        img = Image.open(img_name).convert('RGB')
        
        ret = []
        x1 = self.transform(img)
        ret.append(x1)
        if not self.test:
            x2 = self.transform(img)
            ret.append(x2)
        
        # Add two extra channles if there's only one depth
        if x1.size()[0] == 1:
            print(x1.shape)
            x1 = torch.stack([x1, x1, x1], dim=0).squeeze()
            x2 = torch.stack([x2, x2, x2], dim=0).squeeze()
        elif x1.size()[0] == 2:
            print(f'X equals 2 : {x1.shape}')
            x1 = torch.stack([x1, torch.zeros(1, x1.size()[1], x1.size()[2])], dim=0)
            x2 = torch.stack([x2, torch.zeros(1, x2.size()[1], x2.size()[2])], dim=0)
        return ret, label

#root, imb_type='exp', rand_number=0, train=True, transform=None, target_transform=None, 
#split_num=1, imb_ratio=100, download=False 

class memoboosted_CIFAR10(ImbalanceCIFAR10_index):
    def __init__(self, root, args, train=True, transform=None, target_transform=None, 
                split_num=1, imb_ratio=100, download=False, ):
        super(memoboosted_CIFAR10, self).__init__(root, train=train, transform=transform, 
            target_transform=target_transform, imb_ratio=imb_ratio, download=download) 

        self.root = root 
        self.args = args
        self.momentum_weight = np.empty(self.data.shape[0])
        self.momentum_weight[:] = 0

    def calculate_momentum_weight(self, momentum_loss, epoch):
        momentum_weight = ((momentum_loss[epoch-1]-torch.mean(momentum_loss[epoch-1,:]))/torch.std(momentum_loss[epoch-1,:]))
        momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight[:]))) / 2+1/2).detach().cpu().numpy()
        return momentum_weight

    def update_momentum_weight(self, momentum_loss, epoch):
        momentum_weight_norm = self.calculate_momentum_weight(momentum_loss, epoch)
        self.momentum_weight = momentum_weight_norm

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.args.rand_k == 1:
            min_strength = 10
            memo_boosted_aug = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    RandAugment_prob(self.args.rand_k, min_strength + (self.args.rand_strength - min_strength) * self.momentum_weight[idx], 1.0*self.momentum_weight[idx]),
                    transforms.ToTensor(),
                ])
        imgs = [memo_boosted_aug(img), memo_boosted_aug(img)] 

        return imgs, label, idx

if __name__ == '__main__':
    import argparse
    pp = argparse.ArgumentParser()
    pp.add_argument('--rand_k', default=1)
    pp.add_argument('--rand_strength', default=30)
    args = pp.parse_args()

    data = memoboosted_CIFAR10('../datasets', 
            args,
            split_num=1, 
            imb_ratio=100

        )
    data.__getitem__(0)
