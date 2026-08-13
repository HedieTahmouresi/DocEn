"""Image evaluation metrics (PSNR, SSIM) from scratch.

Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
between predicted document images and ground truth clean targets.
"""

import math
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

    # Compute MSE per image in batch
    mse = torch.mean((pred_f32 - target_f32) ** 2, dim=[-3, -2, -1])

    # Guard against identical images (MSE = 0 -> PSNR = inf)
    psnr_vals = []
    for m in mse:
        val = m.item()
        if val <= eps:
            psnr_vals.append(100.0)  # Standard high cap for identical images
        else:
            psnr = 10.0 * math.log10((data_range ** 2) / val)
            psnr_vals.append(psnr)

    return float(sum(psnr_vals) / len(psnr_vals))


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean SSIM index for a batch [N, C, H, W] in [0, 1]."""
    val = ssim(pred, target, size_average=True)
    return float(val.item())
