"""Sobel edge loss implementation in PyTorch (ADR-006).

Computes the L1 distance between the Sobel gradient magnitudes of predicted
and target document images to penalize blurriness and preserve sharp text edges.
Uses fixed non-trainable Conv2d kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    """Sobel edge magnitude loss using fixed non-trainable 3x3 kernels."""

    def __init__(self, channels: int = 3, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # Define 3x3 Sobel filters for X and Y directions
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

        # Register non-trainable buffers
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude for each channel."""
        # Ensure float32 for stability
        x_f32 = x.to(torch.float32)
        grad_x = F.conv2d(x_f32, self.kernel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x_f32, self.kernel_y, padding=1, groups=self.channels)
        mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        return mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss on Sobel gradient magnitudes."""
        pred_mag = self._gradient_magnitude(pred)
        target_mag = self._gradient_magnitude(target)
        return F.l1_loss(pred_mag, target_mag)
