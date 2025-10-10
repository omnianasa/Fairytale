import torch.nn as nn


from Utils.VGGFeatureExtractor import VGGFeatureExtractor
from Utils.utils import gram_matrix

class StyleLossGram(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGGFeatureExtractor().to(device)
        self.criterion = nn.MSELoss()
        # layers to use: relu1_1, relu2_1, relu3_1
        self.layer_idx = [0,1,2]  # indices in VGGFeatureExtractor output
    
    def forward(self, pred, style):
        pred_feats = self.vgg(pred)
        style_feats = self.vgg(style)
        loss = 0.0
        for idx in self.layer_idx:
            Gp = gram_matrix(pred_feats[idx])
            Gs = gram_matrix(style_feats[idx])
            loss = loss + self.criterion(Gp, Gs)
        return loss