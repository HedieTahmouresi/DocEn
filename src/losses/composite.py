"""Composite loss variants for document enhancement (ADR-006, [REQ-45]).

Implements the four loss comparison variants:
- L-A: MSE (type: 'mse')
- L-B: L1 (type: 'l1')
- L-C: alpha * (1 - MS-SSIM) + (1 - alpha) * L1, default alpha = 0.84 (type: 'l1_msssim')
- L-D: L-C + sobel_weight * SobelL1, default sobel_weight = 0.1 (type: 'l1_msssim_sobel')

Reference:
Zhao et al., "Loss Functions for Image Restoration With Neural Networks" (arXiv:1511.08861)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.ssim import ms_ssim
from src.losses.sobel import SobelLoss


class EnhancementLoss(nn.Module):
    """Configurable loss module for document enhancement ablation."""

    def __init__(
        self,
        loss_type: str = "l1_msssim",
        alpha: float = 0.84,
        sobel_weight: float = 0.1,
        channels: int = 3,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.alpha = alpha
        self.sobel_weight = sobel_weight

        allowed_types = {"mse", "l1", "l1_msssim", "l1_msssim_sobel"}
        if self.loss_type not in allowed_types:
            raise ValueError(f"Unknown loss_type: '{loss_type}'. Must be one of {allowed_types}")

        if "sobel" in self.loss_type:
            self.sobel_fn = SobelLoss(channels=channels)
        else:
            self.sobel_fn = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between pred and target tensors [N, 3, H, W] in [0, 1]."""
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)

        if self.loss_type == "l1":
            return F.l1_loss(pred, target)

        l1_val = F.l1_loss(pred, target)
        ms_ssim_val = ms_ssim(pred, target, size_average=True)
        # Sign check: MS-SSIM is a similarity in [0, 1]; loss is 1 - MS-SSIM
        ms_ssim_loss = 1.0 - ms_ssim_val

        l_c = self.alpha * ms_ssim_loss + (1.0 - self.alpha) * l1_val

        if self.loss_type == "l1_msssim":
            return l_c

        if self.loss_type == "l1_msssim_sobel":
            sobel_loss_val = self.sobel_fn(pred, target)
            return l_c + self.sobel_weight * sobel_loss_val

        raise RuntimeError(f"Unhandled loss type: {self.loss_type}")
