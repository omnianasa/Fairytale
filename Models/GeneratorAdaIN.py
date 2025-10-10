import torch.nn as nn
import torch

from Utils.MLPMapping import MLPMapping
from Utils.utils import conv_block
from Utils.ResBlock import ResBlock
from Utils.AdaIN import AdaIN 



class GeneratorAdaIN(nn.Module):
    def __init__(self, style_dim=256, channels=64, n_res=6):
        super().__init__()
        self.style_dim = style_dim
        # encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, channels, 7,1,3), nn.ReLU(inplace=True))
        self.enc2 = conv_block(channels, channels*2, 4,2,1) #128
        self.enc3 = conv_block(channels*2, channels*4, 4,2,1) #64
        
        # residuals (we will inject AdaIN in residuals)
        self.res_blocks = nn.ModuleList([ResBlock(channels*4) for _ in range(n_res)])
        # for each res block we prepare gamma/beta projection sizes
        # we'll produce gamma,beta per channel for each res block via MLPs
        self.num_adain_layers = n_res
        self.adain_mlps = nn.ModuleList([MLPMapping(style_dim, (channels*4)*2, hidden=512, n_layers=3) for _ in range(self.num_adain_layers)])
        self.adain = AdaIN()
        
        # decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(channels*4, channels*2, 4,2,1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(channels*2, channels, 4,2,1), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(channels, 3, 7,1,3), nn.Tanh())
        
        # style encoder (small convnet) to get style vector from cartoon image
        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7,1,3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4,2,1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4,2,1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, style_dim),
            nn.ReLU(True)
        )
    
    def forward(self, x, style_img):
        # x: real image, style_img: cartoon image
        s = self.style_encoder(style_img)  # B x style_dim
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        f = e3
        # apply res blocks with AdaIN each
        for i, rb in enumerate(self.res_blocks):
            # compute gamma,beta from s
            mlp_out = self.adain_mlps[i](s)  # B x (C*2)
            gamma, beta = torch.chunk(mlp_out, 2, dim=1)
            # apply conv block then adain
            res = rb.block[0](f) if hasattr(rb, 'block') else f  # fallback
            # we actually want to pass through the whole block but inject AdaIN: implement custom
            # Simpler: apply rb.block manually to match adain injection points
            # implement: conv->inorm->relu->conv->inorm ; we'll inject adain after first conv
            # but to keep code readable, do a custom path:
            out = f
            # first conv
            out = rb.block[0](out)  # conv
            out = rb.block[1](out)  # instancenorm
            out = rb.block[2](out)  # relu
            # inject AdaIN here
            out = self.adain(out, gamma, beta)
            out = rb.block[3](out)  # conv
            out = rb.block[4](out)  # instancenorm
            f = f + out
        
        d1 = self.dec1(f)
        d2 = self.dec2(d1)
        out = self.out(d2)
        return out, s