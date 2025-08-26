import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode="train"):

        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, mode + "A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, mode + "B") + "/*.*"))

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[index % len(self.files_B)]).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}
    


def get_dataloader(root, mode="train", batch_size=1, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = ImageDataset(root=root, transform=transform, mode=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

#getting the loaders
train_loader = get_dataloader("./data/cycle_gan_dataset", mode="train")
val_loader   = get_dataloader("./data/cycle_gan_dataset", mode="val")
test_loader  = get_dataloader("./data/cycle_gan_dataset", mode="test", shuffle=False)