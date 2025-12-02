import torch
import torch.nn as nn


class DeepVAE(nn.Module):
    """
    Deep Convolutional Variational Autoencoder (VAE) for images.

    Architecture:
        - Encoder: 4 Conv2d layers with ReLU and flatten
        - Latent space: Fully connected layers for mu and logvar
        - Decoder: Fully connected layer followed by 4 ConvTranspose2d layers

    Args:
        latent_dim (int): Dimensionality of the latent space (default: 256)

    Input:
        - Tensor of shape (batch_size, 3, 128, 128)

    Output:
        - Reconstructed image (batch_size, 3, 128, 128)
        - mu: Mean of latent distribution
        - logvar: Log-variance of latent distribution
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * 8 * 8)

        # Decoder
        self.dec = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent distribution parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            tuple: mu and logvar tensors of shape (batch_size, latent_dim)
        """
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.

        Args:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log-variance of latent distribution

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruct images.

        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        h = self.fc_dec(z)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            tuple: Reconstructed image, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
