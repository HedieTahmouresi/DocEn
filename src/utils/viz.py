"""
Standard Visualization Helpers.

Per conventions §8:
- Corner 0 (TL): Red
- Corner 1 (TR): Green
- Corner 2 (BR): Blue
- Corner 3 (BL): Yellow
- Polygon edges drawn in order 0->1->2->3->0
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


# Colors in RGB format
CORNER_COLORS_RGB = [
    (255, 0, 0),    # 0: TL = Red
    (0, 255, 0),    # 1: TR = Green
    (0, 0, 255),    # 2: BR = Blue
    (255, 255, 0)   # 3: BL = Yellow
]

# Colors in BGR format for OpenCV
CORNER_COLORS_BGR = [
    (0, 0, 255),    # 0: TL = Red
    (0, 255, 0),    # 1: TR = Green
    (255, 0, 0),    # 2: BR = Blue
    (0, 255, 255)   # 3: BL = Yellow
]


def draw_corner_overlay(
    img_rgb: np.ndarray,
    gt_corners: np.ndarray,
    pred_corners: Optional[np.ndarray] = None,
    radius: int = 15,
    thickness: int = 3
) -> np.ndarray:
    """
    Draw standard color-coded corner overlay on an RGB image.

    Args:
        img_rgb: (H, W, 3) uint8 RGB array
        gt_corners: (4, 2) float32 array in [TL, TR, BR, BL] order
        pred_corners: Optional (4, 2) float32 predicted corners
        radius: Circle radius for corner markers
        thickness: Line thickness for quad edges

    Returns:
        vis_img: (H, W, 3) uint8 RGB array with overlay
    """
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    # Convert GT corners if normalized [0, 1]
    gt_pts = gt_corners.copy()
    if np.max(gt_pts) <= 1.0:
        gt_pts[:, 0] *= w
        gt_pts[:, 1] *= h

    # Draw GT quad edges (0->1->2->3->0)
    poly_gt = gt_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [poly_gt], isClosed=True, color=(255, 255, 255), thickness=thickness)

    # Draw GT corners (hollow circles or filled with black border)
    for i, pt in enumerate(gt_pts):
        px, py = int(round(pt[0])), int(round(pt[1]))
        color = CORNER_COLORS_RGB[i % 4]
        cv2.circle(vis, (px, py), radius, color, -1)
        cv2.circle(vis, (px, py), radius + 2, (0, 0, 0), 2)
        cv2.putText(vis, str(i), (px + radius + 2, py + radius + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Draw Pred corners if provided
    if pred_corners is not None:
        pred_pts = pred_corners.copy()
        if np.max(pred_pts) <= 1.0:
            pred_pts[:, 0] *= w
            pred_pts[:, 1] *= h

        poly_pred = pred_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [poly_pred], isClosed=True, color=(0, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)

        for i, pt in enumerate(pred_pts):
            px, py = int(round(pt[0])), int(round(pt[1]))
            color = CORNER_COLORS_RGB[i % 4]
            cv2.circle(vis, (px, py), radius - 3, color, -1)
            cv2.circle(vis, (px, py), radius - 3, (255, 255, 255), 2)

    return vis
