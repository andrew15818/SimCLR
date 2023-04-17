import torch
import matplotlib.pyplot as plt
from collections import defaultdict


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

def plot_class_and_category(values, epoch=0,
                                title='Percentage of Class Samples Chosen for Interpolation',
                                xlabel='Epoch',
                                ylabel='Percentage',
                                filename='plot.png'):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle(title)
    plt.title(title) 
    plt.xlabel('Epochs')
    for classid, v in values.items():
        axes[0].plot(v, label=classid)

    # Place the legend on top to avoid it blocking the curves 
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    axes[0].set_title('By class')

    # CIFAR10 categories only
    categories = {'Many': [1, 4],
                  'Medium': [4, 8],
                  'Few': [7, 10]}

    # Group by  class indices into Many, Medium, Few
    for category, classids in categories.items():
        start, end = classids
        avg = np.zeros((1, epoch+1))
        for i in range(start, end):
            avg += np.array(values[i])
        avg /= (end - start)
        axes[1].plot(avg[0], label=category)

    axes[1].set_title('By category') 
    axes[1].legend()
    plt.savefig(f'{filename}.png')
    plt.close(fig)

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

    #def __str__(self):
    #    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #    return fmtstr.format(**self.__dict__)


# Measure some value per class during training
class ClassAverageMeter:
    def __init__(self, args):
        self.epoch_values = defaultdict(int)
        self.cum_values = defaultdict(list)
        self.cum_var = defaultdict(list)
        self.alpha = 0.9
        self.args = args
        
        if self.args.balanced:
            self.sample_counts = [5000] * args.num_classes
        else:
            self.sample_counts = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50] # amount of samples per class

    # Keep track of values within a single epoch
    def add_batch_value(self, values:torch.Tensor, labels:torch.Tensor):
        for l in range(labels.shape[0]):
            self.epoch_values[labels[l].item()] += values[l].item()
    
    def update(self, epoch_labels=None):
        self.sample_counts = epoch_labels if epoch_labels is not None else self.sample_counts
        for key, value in self.epoch_values.items():
            avg_epoch_value = value / self.sample_counts[key]
            self.cum_values[key].append(avg_epoch_value)

            # Get the EMA value
            if len(self.cum_values[key]) > 1:
                val = self.alpha * (avg_epoch_value - self.cum_values[key][-1]) ** 2 + (1-self.alpha) * (self.cum_var[key][-1])
                self.cum_var[key].append(val)
            else:
                self.cum_var[key].append(0)

        # Append 0 if classid is not present
        classes = [i for i in range(self.args.num_classes)]
        for i in set(classes).difference(self.epoch_values.keys()):
            self.cum_values[i].append(0)

        self.epoch_values = defaultdict(int)
    def get_values(self):
        return self.cum_values

    def get_var_ema(self):
        return self.cum_var
