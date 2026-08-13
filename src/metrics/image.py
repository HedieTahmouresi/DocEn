"""Image evaluation metrics (PSNR, SSIM) from scratch.

Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
between predicted document images and ground truth clean targets.
"""

import torch

from src.losses.ssim import ssim


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute mean Peak Signal-to-Noise Ratio (PSNR) in dB for a batch [N, C, H, W] in [0, 1].

    Parameters
    ----------
    pred, target : torch.Tensor
        Tensors in [0, 1].
    data_range : float, default=1.0
    eps : float, default=1e-10
        Epsilon for numerical stability in log10.

    Returns
    -------
    float
        Average PSNR value in dB across the batch.
    """
    pred_f32 = pred.to(torch.float32)
    target_f32 = target.to(torch.float32)

    # Per-image MSE, never batch-pooled: pooling first would let one easy image
    # flatter a batch that contains a failure (evaluation-spec §1).
    mse = torch.mean((pred_f32 - target_f32) ** 2, dim=[-3, -2, -1])

    # Identical images give MSE = 0 -> PSNR = inf; cap at 100 dB, as skimage-style
    # reference implementations do. Computed on-device in one shot rather than with
    # a .item() per image, which forces a GPU sync for every sample of every epoch.
    psnr = 10.0 * torch.log10((data_range ** 2) / mse.clamp(min=eps))
    psnr = torch.where(mse <= eps, torch.full_like(psnr, 100.0), psnr)

    return float(psnr.mean().item())


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean SSIM index for a batch [N, C, H, W] in [0, 1]."""
    val = ssim(pred, target, size_average=True)
    return float(val.item())
