import torch
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils
from tqdm.auto import tqdm
import os
import numpy as np


from Train.Config import Config
from Train.ImagePairDataset import ImagePairDataset

from Models.GeneratorAdaIN import GeneratorAdaIN
from Utils.EMAHelper import EMAHelper

from Losses.PerceptualLoss import PerceptualLoss
from Losses.SobelEdgeLoss import SobelEdgeLoss
from Losses.StyleLossGram import StyleLossGram



class TrainerStage1:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device

        # dataset + loader
        ds = ImagePairDataset(cfg.real_dir, cfg.cartoon_dir, size=cfg.img_size,
                              max_real=cfg.max_real, max_cartoon=cfg.max_cartoon)
        self.loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, pin_memory=True)

        # models
        self.net = GeneratorAdaIN(style_dim=cfg.style_dim).to(self.device)

        # losses
        self.content_loss = PerceptualLoss(self.device)
        self.style_loss   = StyleLossGram(self.device)
        self.edge_loss_module = SobelEdgeLoss().to(self.device)

        # optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, betas=(0.5,0.999))

        # amp scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device=="cuda"))

        # EMA
        self.ema = EMAHelper(self.net, decay=cfg.ema_decay)

        # bookkeeping
        self.step = 0

        # cartoonGAN-like loss weights
        self.w_content = 1.0
        self.w_style   = 5.0
        self.w_edge    = 50.0

        # warm-up: only edge + content for first few epochs
        self.warmup_epochs = 1

    def save_checkpoint(self, epoch):
        p = os.path.join(self.cfg.save_dir, f"stage1_gen_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": self.net.state_dict(),
            "opt_state": self.opt.state_dict(),
            "ema_shadow": self.ema.shadow,
            "step": self.step
        }, p)
        print("Saved checkpoint:", p)

    def sample_and_save(self, real, cartoon, fake, epoch, tag="sample"):
        def denorm(x):
            x = x.detach().cpu() * 0.5 + 0.5
            return x.clamp(0,1)
        grid = torch.cat([real, cartoon, fake], dim=0)
        grid = denorm(grid)
        path = os.path.join(self.cfg.samples_dir,
                            f"{tag}_epoch{epoch}_step{self.step}.png")
        tv_utils.save_image(grid, path, nrow=real.size(0))

    def train(self):
        print("Start training Stage1 on device:", self.device)
        for epoch in range(1, self.cfg.epochs+1):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}/{self.cfg.epochs}")
            total_loss = 0.0
            for real, cartoon in pbar:
                real = real.to(self.device)
                cartoon = cartoon.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
                    fake, _ = self.net(real, cartoon)

                    # content
                    Lc = self.content_loss(fake, real)
                    # edge consistency
                    Ledge = self.edge_loss_module(fake, cartoon)

                    if epoch <= self.warmup_epochs:
                        # warm-up: ignore style in first epoch(s)
                        loss = self.w_content*Lc + self.w_edge*Ledge
                    else:
                        # full loss after warm-up
                        Ls = self.style_loss(fake, cartoon)
                        loss = (self.w_content*Lc +
                                self.w_style*Ls +
                                self.w_edge*Ledge)

                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                # EMA update
                self.ema.update()

                self.step += 1
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                if self.step % 500 == 0:
                    self.sample_and_save(real, cartoon, fake, epoch, tag="iter_sample")

            avg_loss = total_loss / len(self.loader)
            print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

            if epoch % self.cfg.save_every == 0:
                self.save_checkpoint(epoch)
                # sample with EMA
                self.ema.apply_shadow()
                with torch.no_grad():
                    real0, cartoon0 = next(iter(self.loader))
                    real0 = real0.to(self.device)
                    cartoon0 = cartoon0.to(self.device)
                    fake0, _ = self.net(real0, cartoon0)
                    self.sample_and_save(real0, cartoon0, fake0, epoch, tag="ema_sample")
                self.ema.restore()


if __name__ == "__main__":
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.samples_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer = TrainerStage1(cfg)
    trainer.train()