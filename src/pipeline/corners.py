"""Corner Detection Inference Pipeline.

Fulfills [REQ-32] (preprocess -> predict -> map coordinates back to original resolution -> overlay visualization)
and conventions §8 (fixed corner colors: TL red, TR green, BR blue, BL yellow).
Handles EXIF rotation, greyscale images, and arbitrary aspect ratios.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from src.data.heatmaps import extract_corners_from_heatmaps
from src.models.corner_net import CornerRegNet, CornerHeatmapNet
from src.utils.config import load_config
from src.utils.io import load_image


# Fixed corner color constants per conventions §8
# RGB order: TL red (255,0,0), TR green (0,255,0), BR blue (0,0,255), BL yellow (255,255,0)
CORNER_COLORS_RGB = [
    (255, 0, 0),    # 0: TL - Red
    (0, 255, 0),    # 1: TR - Green
    (0, 0, 255),    # 2: BR - Blue
    (255, 255, 0),  # 3: BL - Yellow
]

CORNER_LABELS = ["TL", "TR", "BR", "BL"]


def preprocess_image_for_corner_model(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    """Preprocess raw image for corner model inference.

    Handles greyscale conversion, resizing to target_size, normalization, and tensor conversion.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image array.
        target_size: (w, h) model input resolution (default: 512, 512).
        mean: Optional channel means for standardization.
        std: Optional channel stds for standardization.

    Returns:
        input_tensor: [1, 3, target_h, target_w] float32 tensor.
    """
    if img_rgb.ndim == 2:
        # Convert greyscale to 3-channel RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 4:
        # Convert RGBA to RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Convert uint8 HWC -> float32 CHW in [0, 1]
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div_(255.0)

    # Standardize input if mean and std are provided
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t

    return tensor.unsqueeze(0)


def predict_corners_from_image(
    model: torch.nn.Module,
    arch: str,
    img_rgb: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict 4 corner coordinates in original image resolution.

    Executes full pipeline: preprocess -> predict -> map back to original resolution.

    Args:
        model: Trained CornerRegNet or CornerHeatmapNet instance.
        arch: Architecture type ("corner_reg" or "corner_heatmap").
        img_rgb: (H_orig, W_orig, 3) uint8 RGB image array.
        target_size: Working model resolution (512, 512).
        mean: Optional channel means for standardization.
        std: Optional channel stds for standardization.
        device: Torch device (CPU or CUDA).

    Returns:
        corners_orig_px: (4, 2) float32 array [x, y] in original image pixel space.
        confidences: (4,) float32 peak confidence scores in [0, 1].
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model.eval()

    orig_h, orig_w = img_rgb.shape[:2]

    input_tensor = preprocess_image_for_corner_model(
        img_rgb, target_size=target_size, mean=mean, std=std
    ).to(device)

    with torch.no_grad():
        if arch == "corner_reg":
            preds = model(input_tensor)  # [1, 8]
            coords_norm = preds.cpu().numpy().reshape(4, 2)
            confidences = np.ones(4, dtype=np.float32)
        else: # "corner_heatmap"
            preds = model(input_tensor)  # [1, 4, 512, 512]
            coords_norm, conf = extract_corners_from_heatmaps(
                preds.cpu().numpy(), window_size=11, normalize=True
            )
            coords_norm = coords_norm.reshape(4, 2)
            confidences = conf.reshape(4)

    # Map normalized coordinates [0, 1] back to original image dimensions (W_orig, H_orig)
    corners_orig_px = coords_norm.copy()
    corners_orig_px[:, 0] *= float(orig_w)
    corners_orig_px[:, 1] *= float(orig_h)

    return corners_orig_px, confidences


def visualize_corner_overlay(
    img_rgb: np.ndarray,
    corners_px: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    line_thickness: int = 3,
    circle_radius: int = 8,
) -> np.ndarray:
    """Render corner keypoints and quadrilateral overlay on raw image per conventions §8.

    Color scheme: TL red, TR green, BR blue, BL yellow.
    Edges drawn in order 0 -> 1 -> 2 -> 3 -> 0.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image array.
        corners_px: (4, 2) float32 corner coordinates in pixel units.
        confidences: Optional (4,) corner confidence scores.
        line_thickness: Polygon line thickness in pixels.
        circle_radius: Keypoint circle radius in pixels.

    Returns:
        overlay_rgb: (H, W, 3) uint8 RGB image with corner overlay rendered.
    """
    canvas = img_rgb.copy()

    # Draw quadrilateral edges 0->1->2->3->0
    pts_int = np.round(corners_px).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_int], isClosed=True, color=(255, 255, 255), thickness=line_thickness + 2)
    cv2.polylines(canvas, [pts_int], isClosed=True, color=(0, 255, 255), thickness=line_thickness)

    # Draw colored corner keypoint circles
    for i in range(4):
        xc, yc = int(round(corners_px[i, 0])), int(round(corners_px[i, 1]))
        color = CORNER_COLORS_RGB[i]

        # White outline circle
        cv2.circle(canvas, (xc, yc), circle_radius + 2, (255, 255, 255), -1)
        # Filled colored circle
        cv2.circle(canvas, (xc, yc), circle_radius, color, -1)

        # Label text
        label_str = CORNER_LABELS[i]
        if confidences is not None:
            label_str += f" ({confidences[i]*100:.0f}%)"

        cv2.putText(
            canvas,
            label_str,
            (xc + 12, yc + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label_str,
            (xc + 12, yc + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


class CornerPipeline:
    """High-level corner detection inference pipeline [REQ-32]."""

    def __init__(self, checkpoint_path: Union[str, Path], config_file: Optional[str] = None, device: Optional[torch.device] = None):
        self.ckpt_path = Path(checkpoint_path)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.arch = ckpt.get("arch", "corner_reg")
        self.cfg = ckpt.get("config", {})

        if config_file:
            loaded_cfg = load_config(exp_file=config_file)
            self.cfg.update(loaded_cfg)

        model_cfg = self.cfg.get("model", {})
        base_ch = model_cfg.get("base_channels", 64)
        levels = model_cfg.get("levels", 4)
        dropout = model_cfg.get("dropout", 0.0)

        # Rebuild the architecture the checkpoint was trained with, not today's defaults.
        # A config that predates `spatial_pool` / `head_norm` is a pre-Phase-07 Approach A
        # checkpoint: average pooling, no BatchNorm1d. `head_norm` changes the parameter
        # count, so guessing wrong raises in load_state_dict rather than failing quietly.
        allow_dropout = bool(model_cfg.get("allow_dropout", dropout > 0.0))

        if self.arch == "corner_reg":
            self.model = CornerRegNet(
                base_channels=base_ch, levels=levels, dropout=dropout,
                allow_dropout=allow_dropout,
                spatial_pool=model_cfg.get("spatial_pool", "avg"),
                head_norm=model_cfg.get("head_norm", False),
            )
        else:
            upsample = model_cfg.get("upsample", "transpose")
            self.model = CornerHeatmapNet(
                base_channels=base_ch, levels=levels, upsample=upsample,
                dropout=dropout, allow_dropout=allow_dropout,
            )

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.mean = self.cfg.get("normalization", {}).get("mean")
        self.std = self.cfg.get("normalization", {}).get("std")

    def predict(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 4 corners in original image space from image path or array.

        Returns:
            corners_orig_px: (4, 2) float32 corner coordinates in original image pixels.
            confidences: (4,) float32 confidence scores.
        """
        if isinstance(image_input, (str, Path)):
            img_rgb = load_image(image_input)
        else:
            img_rgb = image_input

        return predict_corners_from_image(
            model=self.model,
            arch=self.arch,
            img_rgb=img_rgb,
            target_size=(512, 512),
            mean=self.mean,
            std=self.std,
            device=self.device,
        )

    def predict_and_visualize(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict corners and return overlay visualization.

        Returns:
            corners_orig_px: (4, 2) corners in original pixels.
            confidences: (4,) confidence scores.
            overlay_rgb: (H, W, 3) uint8 image with corner overlay rendered.
        """
        if isinstance(image_input, (str, Path)):
            img_rgb = load_image(image_input)
        else:
            img_rgb = image_input

        corners_px, confidences = self.predict(img_rgb)
        overlay = visualize_corner_overlay(img_rgb, corners_px, confidences=confidences)
        return corners_px, confidences, overlay
