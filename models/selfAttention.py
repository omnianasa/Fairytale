import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Self-Attention layer for 2D feature maps.

    Implements the attention mechanism:
        - Computes query, key, and value projections
        - Applies scaled dot-product attention
        - Adds residual connection with learnable scaling (gamma)

    Args:
        in_dim (int): Number of input feature channels

    Input:
        - Tensor of shape (batch_size, in_dim, H, W)

    Output:
        - Tensor of shape (batch_size, in_dim, H, W)
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output feature map with self-attention applied
        """
        B, C, H, W = x.size()

        # Compute query, key, value projections
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        proj_key = self.key_conv(x).view(B, -1, H * W)                        # (B, C//8, H*W)
        proj_value = self.value_conv(x).view(B, C, -1)                        # (B, C, H*W)

        # Compute attention map
        energy = torch.bmm(proj_query, proj_key)                               # (B, H*W, H*W)
        attention = self.softmax(energy)

        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, H*W)
        out = out.view(B, C, H, W)

        # Residual connection with learnable scaling
        return self.gamma * out + x
