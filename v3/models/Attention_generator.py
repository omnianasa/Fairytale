import torch
import torch.nn as nn
from models.selfAttention import SelfAttention


class AttentionGenerator(nn.Module):
    """
    Generator network with self-attention for image generation.

    Architecture:
        - 2 convolutional layers with ReLU
        - Self-Attention layer
        - 2 transposed convolutional layers with ReLU and Tanh output

    Input:
        - Tensor of shape (batch_size, 3, H, W)

    Output:
        - Tensor of shape (batch_size, 3, H, W) with values in [-1, 1]
    """

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            SelfAttention(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, H, W)
        """
        return self.main(x)
