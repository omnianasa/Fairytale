import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features
        # layers indices: relu1_1:1, relu2_1:6, relu3_1:11, relu4_1:20
        self.relu1_1 = nn.Sequential(*list(vgg19.children())[:2])
        self.relu2_1 = nn.Sequential(*list(vgg19.children())[:2+4])
        self.relu3_1 = nn.Sequential(*list(vgg19.children())[:2+4+5])
        self.relu4_1 = nn.Sequential(*list(vgg19.children())[:2+4+5+9])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        out1 = self.relu1_1(x)
        out2 = self.relu2_1(x)
        out3 = self.relu3_1(x)
        out4 = self.relu4_1(x)
        return [out1, out2, out3, out4]
