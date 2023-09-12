import torch
from .resnet import resnet18, resnet50

class SimCLR(torch.nn.Module):
    def __init__(self, base_encoder, encoder_dim=512, proj_hid_dim=128):
        super(SimCLR, self).__init__()
        if base_encoder == 'resnet18':
            self.encoder = resnet18(num_classes=encoder_dim)
        elif base_encoder == 'resnet50':
            self.encoder = resnet50(num_classes=encoder_dim)

        self.projector = torch.nn.Sequential(
                torch.nn.Linear(encoder_dim, proj_hid_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(proj_hid_dim, proj_hid_dim)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x
        
