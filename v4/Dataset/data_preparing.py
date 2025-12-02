from torch.utils.data import Dataset
from torchvision import transforms


class HFIslamicArtDataset(Dataset):
    """
    PyTorch Dataset for Islamic art images from Hugging Face datasets.

    Args:
        hf_dataset_split (Dataset): Hugging Face dataset split (train/test/validation).
        img_size (int): Target size to resize images (default: 128).
        augment (bool): Whether to apply data augmentation (default: True).

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the transformed image at the given index.
    """

    def __init__(self, hf_dataset_split, img_size=128, augment=True):
        self.data = hf_dataset_split
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve and transform the image at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        img = self.data[idx]['img']
        img = img.convert('RGB')
        return self.transform(img)
