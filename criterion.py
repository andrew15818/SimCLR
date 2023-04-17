import torch
import torch.nn as nn
import torch.nn.functional as F

class nt_xent(nn.Module):
    def __init__(self, temperature=0.5):
        super(nt_xent, self).__init__()
        self.temperature = temperature
    
    def _get_mask(self, z1, z2):
        N = 2 * z2.shape[0]
        n = z2.shape[0]
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(n):
            mask[i, n+i] = 0
            mask[n+i, i] = 0
            
        return mask
    
    # z1,z2: [n x d]
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        n = z1.shape[0] # batch size
        
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.t()) / self.temperature
        
        mask = self._get_mask(z1, z2)
        sim_ij = sim.diag(n)
        sim_ji = sim.diag(-n) 
        
        positives = torch.cat([sim_ij, sim_ji], dim=0).unsqueeze(1)
        negatives = sim[mask.bool()].view(2*n, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0]).to(logits.device).long()
        
        return logits, labels 
if __name__ == '__main__':
 
    z1, z2  = torch.rand((64, 128)), torch.rand((64, 128))
    criterion = nt_xent()
    criterion(z1, z2)
    pass
