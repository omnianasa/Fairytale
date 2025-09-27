import torch
import torch.nn as nn
import torch.nn.functional as F
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        """
        encodes the style of an image by measuring the correlations
        between feature maps 
        """
        a, b, c, d = input.size() #[batch, channels, height, width]
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t()) #G = features × featuresᵀ
        return G.div(a * b * c * d) 

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
