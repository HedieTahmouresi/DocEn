"""
COCO Annotation Parsing and Corner Ordering Utility.

Parses RoboFlow COCO polygon segmentation JSON export into ordered (4, 2) float32 arrays
in TL, TR, BR, BL order (conventions §1).

Corner order:
0: TL (Top-Left)
1: TR (Top-Right)
2: BR (Bottom-Right)
3: BL (Bottom-Left)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def cross_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """2D vector cross product (z-component)."""
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def is_convex_quad(pts: np.ndarray) -> bool:
    """
    Check if 4 ordered points form a convex quadrilateral using cross-product sign test.
    pts: (4, 2) float32
    """
    edges = pts[(np.arange(4) + 1) % 4] - pts
    crosses = [cross_2d(edges[i], edges[(i + 1) % 4]) for i in range(4)]
    return all(c > 0 for c in crosses) or all(c < 0 for c in crosses)


def sort_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 arbitrary polygon vertices into TL, TR, BR, BL (clockwise) order.

    Algorithm:
    1. Compute centroid (cx, cy).
    2. Compute polar angle of each point relative to centroid.
    3. Sort by polar angle starting from top-left quadrant (-135 degrees / -3pi/4).

    Args:
        pts: (4, 2) float32, (x, y) coordinates

    Returns:
        sorted_pts: (4, 2) float32, in [TL, TR, BR, BL] order
    """
    cx, cy = np.mean(pts, axis=0)
    
    # Calculate angles relative to centroid in range [-pi, pi]
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    
    # Sort points clockwise starting from top-left quadrant (~ -3pi/4)
    # Re-anchor angles so top-left is lowest angle: shift by +3pi/4 (or 135 deg)
    shifted_angles = (angles + 3 * np.pi / 4) % (2 * np.pi)
    sorted_indices = np.argsort(shifted_angles)
    
    sorted_pts = pts[sorted_indices].astype(np.float32)
    
    # Assert top-left is first point (x < cx and y < cy or smallest sum)
    # If clockwise orientation yielded reverse order, reverse it
    v01 = sorted_pts[1] - sorted_pts[0]
    v03 = sorted_pts[3] - sorted_pts[0]
    if cross_2d(v01, v03) < 0:
        # Swap indices 1 and 3 to ensure clockwise orientation
        sorted_pts = sorted_pts[[0, 3, 2, 1]]
        
    return sorted_pts


def parse_coco_polygon_annotations(
    json_path: Union[str, Path],
    active_filenames: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Parse RoboFlow COCO JSON polygon annotations.

    Args:
        json_path: Path to COCO JSON file
        active_filenames: Optional list of image filenames to filter by

    Returns:
        annotations: Dict mapping filename -> (4, 2) float32 array in [TL, TR, BR, BL] order
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco.get("images", [])}
    ann_map = {ann["image_id"]: ann for ann in coco.get("annotations", [])}

    parsed = {}
    for img_id, img_info in img_map.items():
        filename = img_info.get("extra", {}).get("name") or img_info.get("file_name")
        if active_filenames and filename not in active_filenames:
            continue

        if img_id not in ann_map:
            continue

        ann = ann_map[img_id]
        seg = ann.get("segmentation", [])
        if not seg or len(seg[0]) < 8:
            continue

        pts = np.array(seg[0][:8], dtype=np.float32).reshape(4, 2)
        sorted_pts = sort_corners_clockwise(pts)

        if not is_convex_quad(sorted_pts):
            raise ValueError(f"Annotation quad for {filename} is non-convex!")

        parsed[filename] = sorted_pts

    return parsed
