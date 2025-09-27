from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class Preprocess:
    def __init__(self, image_size=256):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #scale to [-1,1]
        ])

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def save_image(self, tensor, save_path):
        image = tensor.clone().detach()
        image = image * 0.5 + 0.5  # unnormalize to [0,1]
        image = transforms.ToPILImage()(image)
        image.save(save_path)

#CycleGAN Dataset Splitter with preprocessing
class CycleGANDatasetSplitter:
    def __init__(self, folder_A, folder_B, output_dir, split_ratio=(0.8,0.1,0.1), image_size=256, random_seed=42):
        self.folder_A = Path(folder_A)
        self.folder_B = Path(folder_B)
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.preprocessor = Preprocess(image_size=image_size)
        self.random_seed = random_seed

    def split_and_preprocess(self):
        files_A = sorted(list(self.folder_A.glob("*.jpg")))
        files_B = sorted(list(self.folder_B.glob("*.jpg")))

        # Split train/temp
        train_A, temp_A = train_test_split(files_A, test_size=1 - self.split_ratio[0], random_state=self.random_seed)
        train_B, temp_B = train_test_split(files_B, test_size=1 - self.split_ratio[0], random_state=self.random_seed)

        # Split temp into val/test
        val_ratio_adjusted = self.split_ratio[1] / (self.split_ratio[1] + self.split_ratio[2])
        val_A, test_A = train_test_split(temp_A, test_size=1 - val_ratio_adjusted, random_state=self.random_seed)
        val_B, test_B = train_test_split(temp_B, test_size=1 - val_ratio_adjusted, random_state=self.random_seed)

        # Save preprocessed images
        self._save_split(train_A, "trainA")
        self._save_split(train_B, "trainB")
        self._save_split(val_A, "valA")
        self._save_split(val_B, "valB")
        self._save_split(test_A, "testA")
        self._save_split(test_B, "testB")

        print("Dataset split and preprocessed successfully!")

    def _save_split(self, files, subfolder):
        out_folder = self.output_dir / subfolder
        out_folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            img_tensor = self.preprocessor.load_image(f)
            self.preprocessor.save_image(img_tensor, out_folder / f.name)


splitter = CycleGANDatasetSplitter(
    folder_A="data/A",          # video frames
    folder_B="data/B",          # paintings
    output_dir="cycle_gan_dataset",
    split_ratio=(0.8,0.1,0.1),
    image_size=256
)
splitter.split_and_preprocess()
