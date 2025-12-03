import os
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from datasets import load_dataset
from tqdm import tqdm

from Dataset.data_preparing import HFIslamicArtDataset
from losses.utils import vae_loss, perceptual_loss  
from models.GNN import GNN
from models.Deepvae import DeepVAE
from models.selfAttention import SelfAttention


def train_pipeline(
    epochs: int = 50,
    batch_size: int = 8,
    latent_dim: int = 256,
    img_size: int = 128,
    device: str = 'cuda'
) -> None:
    """
    Training pipeline for DeepVAE + GNN + Self-Attention on Islamic art dataset.

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
    train_dataset = HFIslamicArtDataset(dataset['train'], img_size=img_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    vae = DeepVAE(latent_dim).to(device)  # Stage 1: VAE for global structure
    GNN_model = GNN(input_dim=latent_dim, output_dim=latent_dim).to(device)  # Stage 2: GNN for geometric embeddings
    attention_model = SelfAttention(in_dim=latent_dim).to(device)  # Stage 3: Self-Attention for long-range pattern consistency

    # Optimizers
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-4)
    optimizer_gnn = torch.optim.Adam(GNN_model.parameters(), lr=1e-4)
    optimizer_att = torch.optim.Adam(attention_model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        for imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs = imgs.to(device)

            # Stage 1: VAE forward
            recon, mu, logvar = vae(imgs)
            loss_vae = vae_loss(recon, imgs, mu, logvar)

            # Stage 2: GNN forward
            edge_index = None
            # Flatten the VAE output as features for GNN
            G_out = GNN_model(recon.view(recon.size(0), latent_dim, -1), edge_index)

            # Stage 3: Self-Attention forward
            # Self-Attention requires 2D feature map: (B, C, H, W)
            B = G_out.size(0)
            C = 64  # Number of channels for attention
            H = W = int((G_out.size(1) / C) ** 0.5)
            attention_out = attention_model(G_out.view(B, C, H, W))

            # Total loss: VAE + perceptual on attention output
            loss_total = loss_vae + 0.1 * perceptual_loss(attention_out, imgs)

            # Backward pass
            optimizer_vae.zero_grad()
            optimizer_gnn.zero_grad()
            optimizer_att.zero_grad()
            loss_total.backward()
            optimizer_vae.step()
            optimizer_gnn.step()
            optimizer_att.step()

        # Print losses
        print(f"Epoch [{epoch + 1}/{epochs}] VAE Loss: {loss_vae.item():.4f} | Total Loss: {loss_total.item():.4f}")

        # Save intermediate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            utils.save_image(
                (attention_out + 1) / 2,
                f"samples/sample_{epoch + 1}.png",
                nrow=4
            )

    vae.eval()
    GNN_model.eval()
    attention_model.eval()
    with torch.no_grad():
        z = torch.randn(8, vae.fc_mu.out_features).to(device)
        recon = vae.decode(z)
        generated = GNN_model(recon.view(recon.size(0), latent_dim, -1), edge_index)
        B = generated.size(0)
        attention_out = attention_model(generated.view(B, C, H, W))
        utils.save_image((attention_out + 1) / 2, "final_samples/generated.png", nrow=4)
        print("Saved 8 final generated samples to final_samples/generated.png")
    vae.train()
    GNN_model.train()
    attention_model.train()

    torch.save(vae.state_dict(), "vae.pth")
    torch.save(GNN_model.state_dict(), "gnn.pth")
    torch.save(attention_model.state_dict(), "self_attention.pth")
    print("Training complete!")


# RUN
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_pipeline(
        epochs=2500,
        batch_size=12,
        latent_dim=256,
        img_size=128,
        device=device
    )
