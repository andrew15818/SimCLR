import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def save_checkpoint(
        state, isBest=False ,
        filename='checkpoint.pth.tar'):
    torch.save(state, filename)

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

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

def save_checkpoint(state_dict, save_dir, filename):
    if not os.path.exists(save_dir):
        print(f'Making directory {save_dir}')
        os.makedirs(save_dir)
    print(f'Saving file at {os.path.join(save_dir, filename)}')
    torch.save(state_dict, os.path.join(save_dir, filename))
 
def plot(values,
        title='Loss',
        ylabel='Loss',
        filename='imgs/losses.png'):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.plot(values, 'b')
    plt.savefig(filename)
    plt.close(fig)

def plot_by_class(norms, epoch=0, 
                 title='Magnitude of view difference',
                 xlabel='Epochs',
                 ylabel='Magnitude',
                 filename='imgs/difference_norm.png'):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for classid, n in norms.items():
        if classid == 'mean':
            plt.plot(norms['mean'], 'r--', label='mean', linewidth=2, alpha=0.8)
        plt.plot(n , label=classid)
        
    plt.legend()
    plt.savefig(f'{filename}')
    plt.close(fig)

# Plot only by Many, Medium, Few
def plot_category(values, epoch=0,
                    title='Norms',
                    xlabel='Epochs',
                    ylabel='Magnitude',
                    filename='plot.png'):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
     # Just split into thirds
    class_num = len(values.keys())
    interval = class_num // 3
    # Assuming the classes are **sorted** in decreasing sample number
    categories = {'Many': [0,interval],
                  'Medium': [interval, 2*interval +1],
                  'Few': [2*interval+1, class_num]}

    # Group by  class indices into Many, Medium, Few
    for category, bounds in categories.items():
        start, end = bounds
        category_values = list(values.keys())[start:end]
        avg = np.zeros((epoch))
        for classid in category_values:
            avg += values[classid] 
        avg /= (end - start)
        plt.plot(avg, label=category)

    plt.legend()
    plt.savefig(f'{filename}')
    plt.close(fig)

def plot_class_and_category(values, epoch=0,
                                title='Norms',
                                xlabel='Epoch',
                                ylabel='Magnitude',
                                filename='plot.png',
                                useLegend=True):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle(title)
    plt.title(title) 
    plt.xlabel('Epochs')
    for classid, v in values.items():
        axes[0].plot(v, label=classid)

    # Place the legend on top to avoid it blocking the curves 
    if useLegend:
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    axes[0].set_title('By class')

    # Just split into thirds
    class_num = len(list(values.keys()))
    interval = class_num // 3
    categories = {'Many': [0,interval],
                  'Medium': [interval, 2*interval +1],
                  'Few': [2*interval+1, class_num]}

    # Group by  class indices into Many, Medium, Few
    for category, classids in categories.items():
        start, end = classids
        avg = np.zeros((1, epoch+1))
        for i in range(start, end):
            avg += np.array(values[i])
        avg /= (end - start)
        print(end - start)
        axes[1].plot(avg[0], label=category)

    axes[1].set_title('By category') 
    axes[1].legend()
    plt.savefig(f'{filename}')
    plt.close(fig)


# Mahalanobis distance between projection matrices
def mahalanobis_distance(z1, z2, eps=1e-6):
    # Covariance matrix of z1
    inv_cov = torch.inverse(torch.cov(z1))
    
    # Mean difference between projections
    mean_diff = z1 - z2
    dist = torch.sqrt(torch.sum(torch.matmul(mean_diff.T, inv_cov)*mean_diff.T, dim=1))
    print(dist)
    return dist


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# Measure some value per class during training
class ClassAverageMeter:
    def __init__(self, args, sample_counts:dict):
        self.epoch_values = defaultdict(int)
        self.cum_values = defaultdict(list)
        self.cum_var = defaultdict(list)
        self.alpha = 0.9
        self.args = args
        
        self.sample_counts = sample_counts # Amount of images per class
        print(self.sample_counts)
    # Keep track of values within a single epoch
    def add_batch_value(self, values:torch.Tensor, labels:torch.Tensor):
        for l in range(labels.shape[0]):
            self.epoch_values[labels[l].item()] += values[l].item()
    
    def update(self ):
        for key, value in self.epoch_values.items():
            avg_epoch_value = value / self.sample_counts[key]
            self.cum_values[key].append(avg_epoch_value)

            ## Get the EMA value
            #if len(self.cum_values[key]) > 1:
            #    val = self.alpha * (avg_epoch_value - self.cum_values[key][-1]) ** 2 + (1-self.alpha) * (self.cum_var[key][-1])
            #    self.cum_var[key].append(val)
            #else:
            #    self.cum_var[key].append(0)

        # Append 0 if classid is not present
        classes = [i for i in range(self.args.num_classes)]
        for i in set(classes).difference(self.epoch_values.keys()):
            self.cum_values[i].append(0)

        self.epoch_values = defaultdict(int)
    def get_values(self):
        sorted_values = {}
        for classid, values in sorted(self.cum_values.items(), key=lambda item: self.sample_counts[item[0]], reverse=True):
            sorted_values[classid] = values
        return sorted_values

    def get_var_ema(self):
        return self.cum_var
