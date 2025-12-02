import os
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from datasets import load_dataset
from tqdm import tqdm

from Dataset.data_preparing import HFIslamicArtDataset
from losses.utils import d_loss_hinge, g_loss_hinge, vae_loss
from models.Attention_generator import AttentionGenerator
from models.Deepvae import DeepVAE
from models.patch_disc import PatchDiscriminator


def train_pipeline(
    epochs: int = 50,
    batch_size: int = 8,
    latent_dim: int = 256,
    img_size: int = 128,
    device: str = 'cuda'
) -> None:
    """
    Training pipeline for DeepVAE + Attention GAN on Islamic art dataset.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        latent_dim (int): Dimension of VAE latent space.
        img_size (int): Target image size for training.
        device (str): 'cuda' or 'cpu'.
    """
    # Create output directories
    os.makedirs("samples", exist_ok=True)
    os.makedirs("final_samples", exist_ok=True)

    # Load dataset
    dataset = load_dataset("adhamelarabawy/islamic_art")
    train_dataset = HFIslamicArtDataset(
        dataset['train'], img_size=img_size, augment=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    vae = DeepVAE(latent_dim).to(device)
    G = AttentionGenerator().to(device)
    D = PatchDiscriminator().to(device)

    # Optimizers
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-4)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4)

    # Training loop
    for epoch in range(epochs):
        for imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs = imgs.to(device)

            # ---- VAE ----
            recon, mu, logvar = vae(imgs)
            loss_vae = vae_loss(recon, imgs, mu, logvar)
            optimizer_vae.zero_grad()
            loss_vae.backward()
            optimizer_vae.step()

            # ---- GAN ----
            recon_detached = recon.detach()  # Detach from VAE graph

            # Discriminator step
            D_real = D(imgs)
            D_fake = D(G(recon_detached))
            loss_D = d_loss_hinge(D_real, D_fake)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Generator step
            G_out = G(recon_detached)
            D_fake_for_G = D(G_out)
            loss_G = g_loss_hinge(D_fake_for_G)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        # Print losses
        print(f"Epoch {epoch + 1}: VAE {loss_vae.item():.4f} GAN {loss_G.item():.4f}")

        # Save intermediate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            utils.save_image(
                (G_out + 1) / 2,
                f"samples/sample_{epoch + 1}.png",
                nrow=4
            )

    # Generate final samples
    vae.eval()
    G.eval()
    with torch.no_grad():
        z = torch.randn(8, vae.fc_mu.out_features).to(device)
        recon = vae.decode(z)
        generated = G(recon)
        utils.save_image((generated + 1) / 2, "final_samples/generated.png", nrow=4)
        print("Saved 8 final generated samples to final_samples/generated.png")
    vae.train()
    G.train()

    # Save model checkpoints
    torch.save(vae.state_dict(), "vae.pth")
    torch.save(G.state_dict(), "G.pth")
    torch.save(D.state_dict(), "D.pth")
    print("Training complete!")


# ==========================
# 6. Run
# ==========================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_pipeline(
        epochs=2500,
        batch_size=12,
        latent_dim=256,
        img_size=128,
        device=device
    )
