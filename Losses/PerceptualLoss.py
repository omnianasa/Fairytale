import torch.nn as nn

from Utils.VGGFeatureExtractor import VGGFeatureExtractor

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGGFeatureExtractor().to(device)
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)
        # use relu4_1 (index 3)
        return self.criterion(pred_feats[3], target_feats[3])