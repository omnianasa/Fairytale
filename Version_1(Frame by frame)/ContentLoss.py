import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """
    compute content loss between the original content image and the generated one
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  #detach it so it does not track gradients

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target) 
        return input