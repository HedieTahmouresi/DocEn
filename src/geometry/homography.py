"""
Homography and Perspective Rectification Utilities.

Conventions:
- Source corners: (4, 2) float32 in [TL, TR, BR, BL] order
- Destination corners: (4, 2) float32 in [TL, TR, BR, BL] order
- Output rectified crop is standard aspect ratio or square
"""

import cv2
import numpy as np
from typing import Tuple


def get_target_corners(width: int, height: int) -> np.ndarray:
    """
    Get standard destination corners for perspective warp:
    0: (0, 0) - TL
    1: (w-1, 0) - TR
    2: (w-1, h-1) - BR
    3: (0, h-1) - BL
    """
    return np.array([
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)]
    ], dtype=np.float32)


def rectify_document(
    img_rgb: np.ndarray,
    corners: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Warp an unrectified document image into a perspective-rectified crop.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image
        corners: (4, 2) float32 corners in [TL, TR, BR, BL] order
        target_size: (target_w, target_h) output dimensions

    Returns:
        rectified_rgb: (target_h, target_w, 3) uint8 RGB image
    """
    target_w, target_h = target_size
    dst_corners = get_target_corners(target_w, target_h)

    # Compute perspective transform matrix
    src_pts = corners.astype(np.float32)
    H_matrix = cv2.getPerspectiveTransform(src_pts, dst_corners)

    # Warp image
    rectified_bgr = cv2.warpPerspective(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        H_matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR
    )

    return cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
