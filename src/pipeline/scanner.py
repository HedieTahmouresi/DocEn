"""End-to-End Document Scanner Inference Pipeline [REQ-40].

Chains:
1. Corner Detection (Approach B - CornerHeatmapNet): predicts 4 page corners (TL -> TR -> BR -> BL).
2. Perspective Rectification: extract 512x512 rectified crop using homography warp.
3. Document Enhancement (L-D - EnhancementNet): restores degraded crop into clean scan.

Fulfills [REQ-29], [REQ-32], [REQ-40], [REQ-46], [REQ-49] and ADR-012.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from src.geometry.homography import validate_quad
from src.geometry.warp import warp_perspective
from src.pipeline.corners import CornerPipeline, visualize_corner_overlay
from src.pipeline.enhance import enhance_document, load_enhancement_model
from src.utils.io import load_image

logger = logging.getLogger(__name__)

DEFAULT_CORNER_CKPT = "runs/exp-10_corner_approach_b/checkpoints/best.pt"
DEFAULT_ENH_CKPT = "runs/exp-008_enh_l1msssim_sobel/checkpoints/best.pt"


def resolve_checkpoint_path(path: Union[str, Path], default_path: str, fallback_paths: list) -> Path:
    """Resolve checkpoint file path with fallback options."""
    p = Path(path) if path else Path(default_path)
    if p.exists():
        return p
    
    # Try default
    p_default = Path(default_path)
    if p_default.exists():
        return p_default
        
    # Try fallbacks
    for fb in fallback_paths:
        p_fb = Path(fb)
        if p_fb.exists():
            return p_fb
            
    raise FileNotFoundError(f"Checkpoint not found at '{p}'. Fallbacks tested: {fallback_paths}")


class EndToEndScannerPipeline:
    """End-to-End Document Scanner Pipeline [REQ-40].
    
    Chains Corner Detection -> Homography Warp -> Document Enhancement.
    """

    def __init__(
        self,
        corner_ckpt: Optional[Union[str, Path]] = None,
        enh_ckpt: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        corner_path = resolve_checkpoint_path(
            corner_ckpt,
            DEFAULT_CORNER_CKPT,
            ["runs/exp-010_corner_approach_b/checkpoints/best.pt", "runs/exp-10_corner_approach_b/checkpoints/inference_best.pt"]
        )
        enh_path = resolve_checkpoint_path(
            enh_ckpt,
            DEFAULT_ENH_CKPT,
            ["runs/exp-008_enh_l1msssim_sobel/checkpoints/best.pt", "runs/exp-005_enh_mse/checkpoints/best.pt"]
        )

        logger.info(f"Loading Corner Detection model from: {corner_path}")
        self.corner_pipeline = CornerPipeline(checkpoint_path=corner_path, device=self.device)

        logger.info(f"Loading Document Enhancement model from: {enh_path}")
        self.enh_model, self.enh_ckpt_dict = load_enhancement_model(str(enh_path), device=self.device)
        self.enh_config = self.enh_ckpt_dict.get("config", {})

    def scan(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        target_crop_size: Tuple[int, int] = (512, 512),
    ) -> Dict[str, Union[np.ndarray, Image.Image]]:
        """Run complete end-to-end scanner pipeline on an input image.

        Args:
            image_input: File path, PIL Image, or NumPy uint8 array.
            target_crop_size: Output resolution for perspective-rectified crop (default 512x512).

        Returns:
            Dict containing:
                - 'original': (H, W, 3) uint8 RGB array of input image.
                - 'corner_overlay': (H, W, 3) uint8 RGB array with corner overlay.
                - 'rectified': (512, 512, 3) uint8 RGB array before enhancement.
                - 'enhanced': (H, W, 3) uint8 RGB array of final clean scan.
                - 'corners_px': (4, 2) float32 corner coordinates in original image space.
                - 'confidences': (4,) float32 confidence scores.
        """
        # 1. Load image and standardize to uint8 RGB numpy array
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input))
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            img_rgb = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
            img_rgb = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            img_rgb = image_input.copy()
            if img_rgb.ndim == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
            elif img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported image shape: {img_rgb.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = img_rgb.shape[:2]

        # 2. Stage 1: Predict 4 page corners in original photo resolution
        corners_orig_px, confidences = self.corner_pipeline.predict(img_rgb)

        # 3. Corner ordering and quad validity safety check
        is_valid_quad = validate_quad(corners_orig_px, canvas_size=(orig_w, orig_h), margin=0.0)
        if not is_valid_quad:
            logger.warning(
                f"Predicted corners failed quad validation check for image size {orig_w}x{orig_h}. "
                f"Corners: {corners_orig_px.tolist()}. Proceeding without silent re-sorting."
            )

        # 4. Generate corner overlay visualization (TL Red, TR Green, BR Blue, BL Yellow)
        corner_overlay = visualize_corner_overlay(img_rgb, corners_orig_px, confidences=confidences)

        # 5. Stage 2: Perspective Rectification
        rectified_crop = warp_perspective(img_rgb, corners_orig_px, target_size=target_crop_size)

        # 6. Stage 3: Document Enhancement (L-D)
        enhanced_np, _ = enhance_document(
            rectified_crop,
            model=self.enh_model,
            config=self.enh_config,
            device=self.device,
        )

        return {
            "original": img_rgb,
            "corner_overlay": corner_overlay,
            "rectified": rectified_crop,
            "enhanced": enhanced_np,
            "corners_px": corners_orig_px,
            "confidences": confidences,
        }
