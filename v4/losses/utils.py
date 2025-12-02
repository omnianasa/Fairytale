import torch
import torch.nn.functional as F
from torchvision import models

# ======================================================
# Perceptual Loss using pretrained VGG16
# ======================================================
vgg = models.vgg16(pretrained=True).features.eval()
for param in vgg.parameters():
    param.requires_grad = False

vgg = vgg.cuda() if torch.cuda.is_available() else vgg


def perceptual_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute perceptual loss between two images using VGG16 features.

    Args:
        x (torch.Tensor): Reconstructed image tensor, normalized in [-1, 1].
        y (torch.Tensor): Target image tensor, normalized in [-1, 1].

    Returns:
        torch.Tensor: Perceptual loss (MSE of VGG features).
    """
    # Convert from [-1,1] to [0,1]
    x = (x + 1) / 2
    y = (y + 1) / 2
    return F.mse_loss(vgg(x), vgg(y))


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute VAE loss with MSE reconstruction, perceptual loss, and KLD.

    Args:
        recon_x (torch.Tensor): Reconstructed images from VAE.
        x (torch.Tensor): Original input images.
        mu (torch.Tensor): Mean from VAE encoder.
        logvar (torch.Tensor): Log-variance from VAE encoder.

    Returns:
        torch.Tensor: Total VAE loss.
    """
    mse = F.mse_loss(recon_x, x)
    perc = perceptual_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return mse + 0.1 * perc + 1e-4 * kld


# ======================================================
# Hinge Loss for GAN
# ======================================================
def d_loss_hinge(D_real: torch.Tensor, D_fake: torch.Tensor) -> torch.Tensor:
    """
    Discriminator hinge loss.

    Args:
        D_real (torch.Tensor): Discriminator predictions on real images.
        D_fake (torch.Tensor): Discriminator predictions on fake images.

    Returns:
        torch.Tensor: Discriminator hinge loss.
    """
    return torch.mean(F.relu(1. - D_real)) + torch.mean(F.relu(1. + D_fake))


def g_loss_hinge(D_fake: torch.Tensor) -> torch.Tensor:
    """
    Generator hinge loss.

    Args:
        D_fake (torch.Tensor): Discriminator predictions on fake images.

    Returns:
        torch.Tensor: Generator hinge loss.
    """
    return -torch.mean(D_fake)
