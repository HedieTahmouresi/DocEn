"""
Image I/O utilities with BGR/RGB boundary handling and EXIF orientation normalization.

Follows project conventions (§3, §10):
- load_image: returns (H, W, 3) uint8 RGB array in [0, 255]
- save_image: takes (H, W, 3) uint8 RGB array, converts to BGR and saves to disk
- Corrects EXIF orientation tags so loaded array orientation matches annotations
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Tuple, Union


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk, handle EXIF rotation, convert BGR->RGB, return uint8 array.

    Args:
        path: Path to image file

    Returns:
        img: (H, W, 3) uint8, RGB, 0-255
    """
    path_str = str(path)
    # Use PIL to read image and automatically transpose based on EXIF orientation
    try:
        pil_img = Image.open(path_str)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception:
        # Fallback to OpenCV if PIL fails
        img_bgr = cv2.imread(path_str)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at {path_str}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image(img_rgb: np.ndarray, path: Union[str, Path]) -> None:
    """
    Convert RGB image to BGR and save to disk with OpenCV.

    Args:
        img_rgb: (H, W, 3) uint8, RGB, 0-255
        path: Output file path
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path_obj), img_bgr)
    if not success:
        raise RuntimeError(f"Failed to write image to {path}")
