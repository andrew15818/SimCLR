import torchvision.models as models
import torch
import torch.nn as nn

from data import get_transforms, ImbalanceCIFAR10
from torch.utils.data import DataLoader
class SimCLR(torch.nn.Module):
    def __init__(
            self, 
            base_encoder, 
            encoder_dim=2048,
            proj_hid_dim=128):
        super(SimCLR, self).__init__()
        self.encoder_dim = encoder_dim
        self.proj_hid_dim = proj_hid_dim
        # Paper, Figure 8, 2048-d encoder output
        self.encoder = base_encoder(num_classes=self.encoder_dim)
        self.encoder.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                                kernel_size=3, stride=1)
        self.encoder.maxpool = nn.Identity()
        
        self.projector = torch.nn.Sequential(
            nn.Linear(self.encoder_dim, proj_hid_dim),
            nn.BatchNorm1d(proj_hid_dim),
            nn.Linear(proj_hid_dim, proj_hid_dim),
            nn.BatchNorm1d(proj_hid_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x

if __name__ == '__main__':
    train_augs, test_augs = get_transforms()
    train_dataset = ImbalanceCIFAR10(root='../../cvdl2022', train=True, transform=train_augs)
    train_loader = DataLoader(train_dataset, batch_size=64)
    model = SimCLR(models.resnet18) 
    model.train()
    for (x1, x2), labels in train_loader:
        out = model(x1)
        print(out.shape)
        break
