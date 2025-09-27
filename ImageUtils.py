import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ImageUtils:
    """
    Utility class for loading, saving, and visualizing images
    """
    def __init__(self, imsize=None):
        self.imsize = imsize
        self.loader = transforms.Compose([transforms.ToTensor()]) #get the tensor of the image
        self.unloader = transforms.ToPILImage()

    def load_image(self, image_path):
        """
        Loading the image with the new size
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = Image.open(image_path).convert("RGB") # to be more accurate that the image is in rgb
        if self.imsize:
            image = image.resize((self.imsize, self.imsize), Image.Resampling.LANCZOS)
        image = self.loader(image).unsqueeze(0) #shape = [1, 3, imsize, imsize]
        return image.to(device, torch.float)
    

    def save_image(self, tensor, path):
        """
        saving the stylized image to the path
        """
        image = tensor.clone().cpu().squeeze(0)
        image = self.unloader(image)
        image.save(path)


    def show_image(self, tensor, title=None):
        """
        Display the image tensor with a title (if provided)
        """
        image = tensor.clone().cpu().squeeze(0)
        image = self.unloader(image)
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()