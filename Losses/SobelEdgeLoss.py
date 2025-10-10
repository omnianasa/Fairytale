import torch.nn as nn
import torch

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32)

        # Convert to conv filters
        self.register_buffer('kx', kernel_x.view(1,1,3,3))
        self.register_buffer('ky', kernel_y.view(1,1,3,3))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # apply per-channel conv grayscale edges
        pred_gray = pred.mean(1, keepdim=True)
        target_gray = target.mean(1, keepdim=True)

        pred_edges_x = torch.nn.functional.conv2d(pred_gray, self.kx, padding=1)
        pred_edges_y = torch.nn.functional.conv2d(pred_gray, self.ky, padding=1)
        target_edges_x = torch.nn.functional.conv2d(target_gray, self.kx, padding=1)
        target_edges_y = torch.nn.functional.conv2d(target_gray, self.ky, padding=1)

        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-6)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)

        return self.l1(pred_edges, target_edges)