import torch.nn as nn


class MLPMapping(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, n_layers=3):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(n_layers-1):
            layers += [nn.Linear(dim, hidden), nn.ReLU(True)]
            dim = hidden
        layers += [nn.Linear(dim, out_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
