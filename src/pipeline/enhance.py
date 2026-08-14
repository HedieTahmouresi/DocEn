"""Standalone Document Enhancement Pipeline.

Fulfills [REQ-29] and Phase 05 Task D.
Transforms an arbitrary degraded, perspective-rectified document crop into a clean scan.
Includes preprocessing, inference, postprocessing (resize back to original size, uint8 conversion),
and visualization.

Robust against edge cases:
- Non-square / odd aspect ratios
- Greyscale JPEG images
- Images with EXIF orientation rotation tags
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from src.models.unet import EnhancementNet
from src.data.normalization import resolve_from_checkpoint, standardize
from src.utils.config import load_config


def load_enhancement_model(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[EnhancementNet, dict]:
    """Load trained EnhancementNet model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to PyTorch checkpoint .pt file.
    device : Union[str, torch.device], default='cpu'

    Returns
    -------
    Tuple[EnhancementNet, dict]
        (model, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    # Instantiate model
    model = EnhancementNet(
        in_ch=3,
        base_channels=config.get("model", {}).get("base_channels", 64),
        levels=config.get("model", {}).get("levels", 4),
        out_ch=3,
        upsample=config.get("model", {}).get("upsample", "transpose"),
        dropout=0.0,
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, ckpt


def preprocess_image_for_enhancement(
    image: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (512, 512),
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess arbitrary input image into a standardized 512x512 tensor.

    Parameters
    ----------
    image : Union[str, Path, np.ndarray, Image.Image]
        Input image path, PIL Image, or NumPy array.
    target_size : Tuple[int, int], default=(512, 512)
        Target model input size (width, height).
    config : Optional[dict]
        Configuration dict containing standardization settings.

    Returns
    -------
    Tuple[torch.Tensor, Tuple[int, int]]
        (preprocessed_tensor [1, 3, H_target, W_target], original_dimensions (W_orig, H_orig))
    """
    # 1. Load image and handle format / EXIF orientation
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image))
        pil_img = ImageOps.exif_transpose(pil_img)  # Handle EXIF rotation tags
        pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, Image.Image):
        pil_img = ImageOps.exif_transpose(image)
        pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, np.ndarray):
        img_np = image.copy()
        if img_np.ndim == 2:  # Greyscale JPEG
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape: {img_np.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    orig_h, orig_w = img_np.shape[:2]
    orig_dims = (orig_w, orig_h)

    # 2. Resize to 512x512 target resolution
    target_w, target_h = target_size
    resized_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 3. Convert to RGB float32 in [0, 1] tensor [1, 3, 512, 512]
    tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 4. Standardize input if configured (ADR-009)
    if config is not None:
        standardize_flag, mean, std = resolve_from_checkpoint(config)
        if standardize_flag:
            tensor = standardize(tensor, mean, std)

    return tensor, orig_dims



def postprocess_enhancement_output(
    output_tensor: torch.Tensor,
    original_dims: Tuple[int, int],
) -> Tuple[np.ndarray, Image.Image]:
    """Postprocess enhancement model output back to original size uint8 RGB image.

    Parameters
    ----------
    output_tensor : torch.Tensor
        Model prediction tensor [1, 3, H, W] in [0, 1].
    original_dims : Tuple[int, int]
        Original image dimensions (W_orig, H_orig).

    Returns
    -------
    Tuple[np.ndarray, Image.Image]
        (enhanced_rgb_numpy uint8 [H_orig, W_orig, 3], enhanced_pil_image)
    """
    # 1. Clamp output to [0, 1] and detach
    out_sq = output_tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    out_np = out_sq.permute(1, 2, 0).numpy()  # (512, 512, 3) float32 in [0, 1]

    # 2. Resize back to original image dimensions (W_orig, H_orig)
    orig_w, orig_h = original_dims
    if (out_np.shape[1], out_np.shape[0]) != (orig_w, orig_h):
        resized_out = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    else:
        resized_out = out_np

    # 3. Convert to uint8 [0, 255]
    uint8_out = (np.clip(resized_out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil_out = Image.fromarray(uint8_out, mode="RGB")

    return uint8_out, pil_out


def enhance_document(
    image: Union[str, Path, np.ndarray, Image.Image],
    checkpoint_path: Optional[str] = None,
    model: Optional[EnhancementNet] = None,
    config: Optional[dict] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, Image.Image]:
    """Run full enhancement pipeline on an arbitrary image.

    Parameters
    ----------
    image : Union[str, Path, np.ndarray, Image.Image]
        Input raw or rectified document image.
    checkpoint_path : Optional[str]
        Path to model checkpoint. Default: runs/exp-008_enh_l1msssim_sobel/checkpoints/best.pt.
    model : Optional[EnhancementNet]
        Pre-loaded model instance.
    config : Optional[dict]
        Optional config dict for normalization.
    device : Union[str, torch.device], default='cpu'

    Returns
    -------
    Tuple[np.ndarray, Image.Image]
        (enhanced_numpy_rgb_uint8, enhanced_pil_image)
    """
    if model is None:
        if checkpoint_path is None:
            checkpoint_path = "runs/exp-008_enh_l1msssim_sobel/checkpoints/best.pt"
            if not Path(checkpoint_path).exists():
                # Fallback to exp-005 if exp-008 is missing
                checkpoint_path = "runs/exp-005_enh_mse/checkpoints/best.pt"
        model, ckpt = load_enhancement_model(checkpoint_path, device=device)
        if config is None:
            config = ckpt.get("config", {})

    # 1. Preprocess
    input_tensor, orig_dims = preprocess_image_for_enhancement(
        image, target_size=(512, 512), config=config
    )
    input_tensor = input_tensor.to(device)

    # 2. Model Predict
    with torch.no_grad():
        pred_tensor = model(input_tensor)

    # 3. Postprocess
    enhanced_np, enhanced_pil = postprocess_enhancement_output(pred_tensor, orig_dims)

    return enhanced_np, enhanced_pil


def save_enhanced_image(enhanced_rgb: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save enhanced RGB numpy array to disk as BGR image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_img = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_img)
