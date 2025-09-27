import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
import matplotlib.pyplot as plt

class CycleGANTrainer:
    def __init__(self, G, F, D_A, D_B, device="cuda"):
        self.device = device

        # Networks
        self.G = G.to(device)
        self.F = F.to(device)
        self.D_A = D_A.to(device)
        self.D_B = D_B.to(device)

        # Losses
        self.adv_criterion = nn.MSELoss()
        self.cycle_criterion = nn.L1Loss()
        self.identity_criterion = nn.L1Loss()

        # Optimizers
        self.optimizer_G = optim.Adam(chain(G.parameters(), F.parameters()), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D_A = optim.Adam(D_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(D_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Loss history for visualization
        self.losses = {"G": [], "D_A": [], "D_B": [], "cycle": [], "identity": []}

    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def evaluate_step(self, real_A, real_B, lambda_cycle=10.0, lambda_id=5.0):
            self.set_input(real_A, real_B)
            with torch.no_grad():
                # Generate fakes
                fake_B = self.G(self.real_A)
                fake_A = self.F(self.real_B)
        
                # Adversarial losses
                valid_B = torch.ones_like(self.D_B(fake_B), device=self.device)
                loss_G_adv = self.adv_criterion(self.D_B(fake_B), valid_B)
        
                valid_A = torch.ones_like(self.D_A(fake_A), device=self.device)
                loss_F_adv = self.adv_criterion(self.D_A(fake_A), valid_A)
                loss_adv = loss_G_adv + loss_F_adv
        
                # Cycle-consistency loss
                recov_A = self.F(fake_B)
                recov_B = self.G(fake_A)
                loss_cycle = self.cycle_criterion(recov_A, self.real_A) + self.cycle_criterion(recov_B, self.real_B)
        
                # Identity loss
                id_A = self.F(self.real_A)
                id_B = self.G(self.real_B)
                loss_identity = self.identity_criterion(id_A, self.real_A) + self.identity_criterion(id_B, self.real_B)
        
                # Total generator loss
                loss_G_total = loss_adv + lambda_cycle * loss_cycle + lambda_id * loss_identity
        
                # Discriminator losses
                pred_real_A = self.D_A(self.real_A)
                pred_fake_A = self.D_A(fake_A)
                valid = torch.ones_like(pred_real_A)
                fake = torch.zeros_like(pred_fake_A)
                loss_D_A = 0.5 * (self.adv_criterion(pred_real_A, valid) + self.adv_criterion(pred_fake_A, fake))
        
                pred_real_B = self.D_B(self.real_B)
                pred_fake_B = self.D_B(fake_B)
                valid = torch.ones_like(pred_real_B)
                fake = torch.zeros_like(pred_fake_B)
                loss_D_B = 0.5 * (self.adv_criterion(pred_real_B, valid) + self.adv_criterion(pred_fake_B, fake))
        
            return loss_G_total.item(), loss_D_A.item(), loss_D_B.item()


    def train_step(self, lambda_cycle=10.0, lambda_id=5.0):

        # === Train Generators G and F ===
        self.optimizer_G.zero_grad()

        # Generate fakes
        fake_B = self.G(self.real_A)
        fake_A = self.F(self.real_B)

        # Adversarial losses
        valid = torch.ones_like(self.D_B(fake_B), device=self.device)
        loss_G_adv = self.adv_criterion(self.D_B(fake_B), valid)

        valid = torch.ones_like(self.D_A(fake_A), device=self.device)
        loss_F_adv = self.adv_criterion(self.D_A(fake_A), valid)

        loss_adv = loss_G_adv + loss_F_adv

        # Cycle-consistency loss
        recov_A = self.F(fake_B)
        recov_B = self.G(fake_A)
        loss_cycle = self.cycle_criterion(recov_A, self.real_A) + self.cycle_criterion(recov_B, self.real_B)

        # Identity loss
        id_A = self.F(self.real_A)
        id_B = self.G(self.real_B)
        loss_identity = self.identity_criterion(id_A, self.real_A) + self.identity_criterion(id_B, self.real_B)

        # Total generator loss
        loss_G_total = loss_adv + lambda_cycle * loss_cycle + lambda_id * loss_identity
        loss_G_total.backward()
        self.optimizer_G.step()

        # === Train Discriminator D_A ===
        self.optimizer_D_A.zero_grad()
        pred_real = self.D_A(self.real_A)
        pred_fake = self.D_A(fake_A.detach())
        valid = torch.ones_like(pred_real)
        fake = torch.zeros_like(pred_fake)
        loss_D_A = 0.5 * (self.adv_criterion(pred_real, valid) + self.adv_criterion(pred_fake, fake))
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # === Train Discriminator D_B ===
        self.optimizer_D_B.zero_grad()
        pred_real = self.D_B(self.real_B)
        pred_fake = self.D_B(fake_B.detach())
        valid = torch.ones_like(pred_real)
        fake = torch.zeros_like(pred_fake)
        loss_D_B = 0.5 * (self.adv_criterion(pred_real, valid) + self.adv_criterion(pred_fake, fake))
        loss_D_B.backward()
        self.optimizer_D_B.step()

        # Save losses
        self.losses["G"].append(loss_adv.item())
        self.losses["cycle"].append(loss_cycle.item())
        self.losses["identity"].append(loss_identity.item())
        self.losses["D_A"].append(loss_D_A.item())
        self.losses["D_B"].append(loss_D_B.item())

        return loss_G_total.item(), loss_D_A.item(), loss_D_B.item()

    def plot_losses(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.losses["G"], label="G adv")
        plt.plot(self.losses["cycle"], label="Cycle")
        plt.plot(self.losses["identity"], label="Identity")
        plt.plot(self.losses["D_A"], label="D_A")
        plt.plot(self.losses["D_B"], label="D_B")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()