import torch
import torch.nn as nn
import torchvision.models as models
import copy
from normalizer import Normalization
from ContentLoss import ContentLoss
from StyleLoss import StyleLoss

class StyleTransferModel:
    """
    Building a style transfer model by inserting ContentLoss and StyleLoss 
    layers into a copy of the VGG19 network
    """
    def __init__(self, style_img, content_layers, style_layers):

        self.style_img = style_img
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def build(self, content_img):
        cnn = copy.deepcopy(self.cnn)
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(self.device)

        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for j in range(len(model) - 1, -1, -1):
            if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
                break

        return model[:j+1], style_losses, content_losses