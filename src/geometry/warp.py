"""Warp and perspective rectification interface [REQ-40]."""

from typing import Tuple
import numpy as np

from src.geometry.homography import rectify_document, compute_homography, get_target_corners, validate_quad


def warp_perspective(
    img_rgb: np.ndarray,
    corners: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Warp an unrectified document image into a perspective-rectified crop.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image array.
        corners: (4, 2) float32 corner coordinates in [TL, TR, BR, BL] order (pixel units).
        target_size: (w, h) dimensions of rectified output image (default 512x512).

    Returns:
        rectified_rgb: (target_h, target_w, 3) uint8 RGB image array.
    """
    return rectify_document(img_rgb, corners, target_size=target_size)
