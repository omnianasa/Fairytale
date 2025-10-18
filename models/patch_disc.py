import torch
import torch.nn as nn
from models.selfAttention import SelfAttention


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator with self-attention.

    Architecture:
        - 2 Conv2d layers with LeakyReLU
        - Self-Attention layer
        - Final Conv2d layer producing a patch-level output

    Input:
        - Tensor of shape (batch_size, 3, H, W)

    Output:
        - Flattened patch-level discriminator output (batch_size * num_patches)
    """

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            SelfAttention(128),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=0)  # PatchGAN output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PatchGAN discriminator.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Flattened patch-level discriminator output
        """
        return self.main(x).view(-1)
