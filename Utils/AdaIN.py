import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x, gamma, beta):
        # x: BxCxHxW, gamma/beta: BxC
        B,C,H,W = x.size()
        mean = x.view(B,C,-1).mean(dim=2).view(B,C,1,1)
        std = x.view(B,C,-1).std(dim=2).view(B,C,1,1) + self.eps
        x_norm = (x - mean)/std
        gamma = gamma.view(B,C,1,1)
        beta = beta.view(B,C,1,1)
        return gamma*x_norm + beta