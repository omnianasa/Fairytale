from torch.utils.data import Dataset
import os
from PIL import Image

from torch.utils.data import DataLoader

import os
from torch.utils.data import Dataset
from PIL import Image

from data.split import Preprocess


class FramesDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames = [
            os.path.join(frames_dir, f) 
            for f in os.listdir(frames_dir) 
            if f.endswith((".jpg", ".png"))
        ]
        self.frames.sort()  # keep chronological order (important for video)
        
        preprocess = Preprocess()
        self.transform = transform if transform else preprocess.transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_path = self.frames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image



