from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# Mostly the pseudocode from paper apendix
def _get_color_distortion(s=1.0):
    # s is the strength
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])
    return color_distort

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        _get_color_distortion(s=0.5),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return train_transforms, test_transforms

# Just a wrapper that returns imgs in same format as ImbalancedCIFAR10
class BalancedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, 
            target_transform=None, download=False):
        super(BalancedCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.transform(img)

        return [img1, img2], target

# Taken from https://github.com/Liuhong99/Imbalanced-SSL/blob/main/cifar/imbalance_cifar.py
class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False, shuffleClasses=True):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.imb_factor = imb_factor
        self.shuffleClasses = shuffleClasses
        self.gen_imbalanced_data(img_num_list)
        self.img_num_list = img_num_list

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        if self.shuffleClasses:
            np.random.shuffle(classes)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
         
        # Save current dataset so we know Many, Medium, Few classes
        np.save(f'splits/cifar{self.cls_num}_imb.npy', np.array(self.targets))

    def get_cls_num_dict(self):
        #cls_num_dict = {}
        #for i in range(self.cls_num):
        #    cls_num_dict[i] = self.num_per_cls_dict[i]
        return self.num_per_cls_dict

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        img = Image.fromarray(img)

        x1 = self.transform(img)
        x2 = self.transform(img)
        return [x1, x2], target

class ImbalanceCIFAR100(ImbalanceCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

if __name__=='__main__':
    trainaugs, testaugs = get_transforms()
    dataset = ImbalanceCIFAR100(root='../datasets', 
                            train=True, 
                            transform=trainaugs)
    print(dataset.get_cls_num_dict())
