import torch
import torch.nn as nn

class Normalization(nn.Module):
    """
    Normalizes an input image using the mean and std values that the 
    VGG network expects. It is the first layer in style transfer model
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std