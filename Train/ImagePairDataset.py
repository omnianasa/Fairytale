import os
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class ImagePairDataset(Dataset):
    def __init__(self, real_root, cartoon_root, size=256, max_real=None, max_cartoon=None):
        super().__init__()
        
        def get_all_images(root):
            exts = (".jpg",".png",".jpeg")
            paths = []
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(exts):
                        paths.append(os.path.join(dirpath, f))
            return sorted(paths)

        self.real_paths = get_all_images(real_root)
        self.cartoon_paths = get_all_images(cartoon_root)

        if max_real:
            self.real_paths = self.real_paths[:max_real]
        if max_cartoon:
            self.cartoon_paths = self.cartoon_paths[:max_cartoon]

        self.real_len = len(self.real_paths)
        self.cartoon_len = len(self.cartoon_paths)
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        assert self.real_len>0 and self.cartoon_len>0, f"Empty dataset paths! Real={self.real_len}, Cartoon={self.cartoon_len}"
    
    def __len__(self):
        return max(self.real_len, self.cartoon_len)
    
    def __getitem__(self, idx):
        r = self.transform(Image.open(self.real_paths[idx % self.real_len]).convert("RGB"))
        c = self.transform(Image.open(self.cartoon_paths[idx % self.cartoon_len]).convert("RGB"))
        return r, c
