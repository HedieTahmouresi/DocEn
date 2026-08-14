"""SSIM and MS-SSIM implementations in PyTorch from scratch (ADR-010).

Calculates Structural Similarity (SSIM) and Multi-Scale SSIM (MS-SSIM)
for 4D tensors (N, C, H, W) in [0, 1].

References:
- Wang et al., "Multi-scale structural similarity for image quality assessment" (2003)
- Zhao et al., "Loss Functions for Image Restoration With Neural Networks" (2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_window_1d(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian kernel normalized to sum to 1."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size // 2)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def create_gaussian_window_2d(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian window for conv2d with groups=channels."""
    g_1d = create_gaussian_window_1d(window_size, sigma)
    g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)  # (window_size, window_size)
    g_2d = g_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    window = g_2d.repeat(channels, 1, 1, 1)
    return window


def _ssim_per_channel(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channels: int,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
    crop_border: bool = True,
):
    """Compute local SSIM map, mean luminance map (l), and mean contrast-structure map (cs)."""
    pad = window_size // 2

    # Choose padding mode based on image dimensions
    if img1.size(-2) > pad and img1.size(-1) > pad:
        mode = 'reflect'
    else:
        mode = 'replicate'

    img1_padded = F.pad(img1, (pad, pad, pad, pad), mode=mode)
    img2_padded = F.pad(img2, (pad, pad, pad, pad), mode=mode)

    mu1 = F.conv2d(img1_padded, window, padding=0, groups=channels)
    mu2 = F.conv2d(img2_padded, window, padding=0, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_padded * img1_padded, window, padding=0, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2_padded * img2_padded, window, padding=0, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1_padded * img2_padded, window, padding=0, groups=channels) - mu1_mu2

    # Clamp variances for numerical stability
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    l_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    ssim_map = l_map * cs_map

    if crop_border and pad > 0 and img1.size(-2) > 2 * pad and img1.size(-1) > 2 * pad:
        ssim_map = ssim_map[:, :, pad:-pad, pad:-pad]
        l_map = l_map[:, :, pad:-pad, pad:-pad]
        cs_map = cs_map[:, :, pad:-pad, pad:-pad]

    return ssim_map, l_map, cs_map


from functools import lru_cache


@lru_cache(maxsize=16)
def _get_cached_gaussian_window_2d(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    return create_gaussian_window_2d(window_size, sigma, channels)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
    crop_border: bool = True,
) -> torch.Tensor:
    """Compute SSIM index between two batches of images [N, C, H, W] in [0, 1].

    Note: Always computes in float32 for numerical stability.
    """
    # Enforce float32 for AMP safety
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)

    channels = img1.size(1)
    window = _get_cached_gaussian_window_2d(window_size, sigma, channels).to(
        device=img1.device, dtype=img1.dtype
    )

    ssim_map, _, _ = _ssim_per_channel(
        img1, img2, window, window_size, channels, C1=C1, C2=C2, crop_border=crop_border
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[-3, -2, -1])


def ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
    weights: torch.Tensor = None,
    crop_border: bool = True,
) -> torch.Tensor:
    """Compute Multi-Scale SSIM (MS-SSIM) between two batches of images [N, C, H, W] in [0, 1].

    Parameters
    ----------
    img1, img2 : torch.Tensor
        Input images [N, C, H, W] float32 in [0, 1].
    window_size : int, default=11
    sigma : float, default=1.5
    size_average : bool, default=True
    C1, C2 : float
        Stability constants for SSIM.
    weights : torch.Tensor, optional
        Weights for 5 scales. Default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] (Zhao et al. 2017).
    crop_border : bool, default=True
        Whether to crop border pixels to match skimage evaluation protocol.

    Returns
    -------
    torch.Tensor
        Scalar MS-SSIM value if size_average=True, else [N] tensor.
    """
    # Enforce float32 for AMP safety (prevents NaN under fp16)
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)

    if weights is None:
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            dtype=torch.float32,
            device=img1.device,
        )
    else:
        weights = weights.to(device=img1.device, dtype=torch.float32)

    levels = weights.size(0)
    channels = img1.size(1)

    window = create_gaussian_window_2d(window_size, sigma, channels).to(
        device=img1.device, dtype=img1.dtype
    )

    mcs_list = []
    l_final = None

    curr_img1 = img1
    curr_img2 = img2

    for i in range(levels):
        ssim_map, l_map, cs_map = _ssim_per_channel(
            curr_img1, curr_img2, window, window_size, channels, C1=C1, C2=C2, crop_border=crop_border
        )

        if i == levels - 1:
            l_final = l_map.mean(dim=[-3, -2, -1])  # per-image mean luminance at last scale
            cs_val = cs_map.mean(dim=[-3, -2, -1])
            mcs_list.append(cs_val)
        else:
            cs_val = cs_map.mean(dim=[-3, -2, -1])
            mcs_list.append(cs_val)

            # Downsample images for next scale
            curr_img1 = F.avg_pool2d(curr_img1, kernel_size=2, stride=2, padding=0)
            curr_img2 = F.avg_pool2d(curr_img2, kernel_size=2, stride=2, padding=0)

    # Stack cs values across scales: shape [levels, N]
    mcs_stack = torch.stack(mcs_list, dim=0)  # (levels, N)
    mcs_stack = torch.clamp(mcs_stack, min=1e-8)
    l_final = torch.clamp(l_final, min=1e-8)

    # MS-SSIM calculation in log domain for stability:
    # MS-SSIM = (l_final ** weights[-1]) * prod(mcs_i ** weights_i)
    log_mcs = torch.log(mcs_stack)
    weights_col = weights.unsqueeze(1)  # (levels, 1)

    overall_ms_ssim = torch.exp(
        torch.sum(weights_col[:-1] * log_mcs[:-1], dim=0) + weights_col[-1] * (log_mcs[-1] + torch.log(l_final))
    )

    if size_average:
        return overall_ms_ssim.mean()
    else:
        return overall_ms_ssim


class SSIM(nn.Module):
    """SSIM loss/metric module."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        size_average: bool = True,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
        crop_border: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.C1 = C1
        self.C2 = C2
        self.crop_border = crop_border

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim(
            img1,
            img2,
            window_size=self.window_size,
            sigma=self.sigma,
            size_average=self.size_average,
            C1=self.C1,
            C2=self.C2,
            crop_border=self.crop_border,
        )


class MSSSIM(nn.Module):
    """MS-SSIM loss/metric module."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        size_average: bool = True,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
        weights: torch.Tensor = None,
        crop_border: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.C1 = C1
        self.C2 = C2
        self.weights = weights
        self.crop_border = crop_border

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ms_ssim(
            img1,
            img2,
            window_size=self.window_size,
            sigma=self.sigma,
            size_average=self.size_average,
            C1=self.C1,
            C2=self.C2,
            weights=self.weights,
            crop_border=self.crop_border,
        )
