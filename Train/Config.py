import torch
import os
import numpy as np

class Config:

    cartoon_dir = "/kaggle/input/cartoon-faces-googles-cartoon-set/cartoonset100k_jpg"
    real_dir = "/kaggle/input/celeba-dataset/img_align_celeba"
    save_dir = "/kaggle/working/str_gan/checkpoints"
    samples_dir = "/kaggle/working/str_gan/samples"

    img_size = 256
    batch_size = 10
    num_workers = 2
    lr = 1e-6
    epochs = 25
    style_dim = 256
    weight_content = 1.0
    weight_style = 10.0
    weight_edge = 2.0
    ema_decay = 0.999
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_every = 1  
    max_real = None 
    max_cartoon = None
    seed = 42

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
os.makedirs(cfg.samples_dir, exist_ok=True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
    