import torch.nn as nn
import torch

def conv_block(in_c, out_c, ks=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, ks, stride=stride, padding=pad),
        nn.InstanceNorm2d(out_c, affine=False),
        nn.ReLU(inplace=True)
    )


def gram_matrix(feat):
    B, C, H, W = feat.shape
    f = feat.view(B, C, H*W)
    return torch.bmm(f, f.transpose(1,2)) / (C*H*W)