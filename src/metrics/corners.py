"""Corner Detection Evaluation Metrics.

Fulfills [REQ-31] (Mean Corner Error in px and % of diagonal, Success Rate @ 1% and 2% thresholds)
and ADR-011 (diagonal reference = 724.1 px @ 512x512).
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch


def compute_corner_errors(
    pred_corners: Union[np.ndarray, torch.Tensor],
    target_corners: Union[np.ndarray, torch.Tensor],
    canvas_size: Tuple[int, int] = (512, 512),
    normalized_input: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Compute per-corner Euclidean distance errors in pixels.

    Args:
        pred_corners: Predicted corners of shape (N, 4, 2) or (N, 8) or (4, 2) or (8,).
        target_corners: Target corners of same shape as pred_corners.
        canvas_size: (W, H) image size in pixels (default: 512, 512).
        normalized_input: If True, input coordinates are in [0, 1] and will be scaled by canvas_size.
                          If False, input coordinates are already in pixel units.

    Returns:
        corner_errors_px: Array/Tensor of shape (N, 4) containing Euclidean error in pixels for each corner.
    """
    is_torch = isinstance(pred_corners, torch.Tensor)
    device = pred_corners.device if is_torch else None

    if is_torch:
        pred_np = pred_corners.detach().cpu().numpy()
        target_np = target_corners.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_corners, dtype=np.float32)
        target_np = np.asarray(target_corners, dtype=np.float32)

    if pred_np.shape != target_np.shape:
        raise ValueError(f"Shape mismatch between pred {pred_np.shape} and target {target_np.shape}")

    orig_ndim = pred_np.ndim
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(1, 4, 2)
        target_np = target_np.reshape(1, 4, 2)
    elif pred_np.ndim == 2:
        if pred_np.shape[1] == 8:
            pred_np = pred_np.reshape(-1, 4, 2)
            target_np = target_np.reshape(-1, 4, 2)
        elif pred_np.shape == (4, 2):
            pred_np = pred_np[np.newaxis, ...]
            target_np = target_np[np.newaxis, ...]

    W, H = canvas_size
    pred_px = pred_np.copy()
    target_px = target_np.copy()

    if normalized_input:
        pred_px[:, :, 0] *= float(W)
        pred_px[:, :, 1] *= float(H)
        target_px[:, :, 0] *= float(W)
        target_px[:, :, 1] *= float(H)

    # Euclidean distance per corner: sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)
    diff = pred_px - target_px
    errors_px = np.linalg.norm(diff, axis=-1)  # (N, 4)

    if orig_ndim <= 2 and (pred_corners.shape == (4, 2) or pred_corners.shape == (8,)):
        errors_px = errors_px[0]

    if is_torch:
        return torch.from_numpy(errors_px).to(device)
    return errors_px


def compute_corner_metrics(
    pred_corners: Union[np.ndarray, torch.Tensor],
    target_corners: Union[np.ndarray, torch.Tensor],
    canvas_size: Tuple[int, int] = (512, 512),
    normalized_input: bool = True,
) -> Dict[str, float]:
    """Compute complete suite of corner detection metrics [REQ-31].

    Metrics computed:
    - mean_corner_error_px: Mean Corner Error in pixels.
    - mean_corner_error_pct: Mean Corner Error as % of diagonal.
    - success_rate_1pct: % of images where all 4 corners have error <= 1% diagonal.
    - success_rate_2pct: % of images where all 4 corners have error <= 2% diagonal.
    - success_rate_1pct_corner: % of individual corners with error <= 1% diagonal.
    - success_rate_2pct_corner: % of individual corners with error <= 2% diagonal.
    - TL_err_px, TR_err_px, BR_err_px, BL_err_px: Per-corner mean errors in pixels.

    Args:
        pred_corners: Predicted corners of shape (N, 4, 2) or (N, 8) or (4, 2) or (8,).
        target_corners: Target corners of same shape as pred_corners.
        canvas_size: (W, H) image size in pixels (default: 512, 512).
        normalized_input: If True, input coordinates are in [0, 1].

    Returns:
        Dict[str, float]: Dictionary of computed metric values.
    """
    errors_px = compute_corner_errors(
        pred_corners=pred_corners,
        target_corners=target_corners,
        canvas_size=canvas_size,
        normalized_input=normalized_input,
    )

    if isinstance(errors_px, torch.Tensor):
        errors_px = errors_px.detach().cpu().numpy()

    if errors_px.ndim == 1:
        errors_px = errors_px[np.newaxis, :]  # (1, 4)

    N = errors_px.shape[0]
    W, H = canvas_size
    diag_px = float(np.sqrt(W ** 2 + H ** 2))  # ~724.0773 px for 512x512

    # Thresholds in pixels
    thresh_1pct_px = 0.01 * diag_px  # ~7.24 px @ 512
    thresh_2pct_px = 0.02 * diag_px  # ~14.48 px @ 512

    mean_err_px = float(np.mean(errors_px))
    mean_err_pct = float((mean_err_px / diag_px) * 100.0)

    # Per-sample max error across the 4 corners
    max_sample_err = np.max(errors_px, axis=1)  # (N,)

    success_1pct_sample = float(np.mean(max_sample_err <= thresh_1pct_px) * 100.0)
    success_2pct_sample = float(np.mean(max_sample_err <= thresh_2pct_px) * 100.0)

    success_1pct_corner = float(np.mean(errors_px <= thresh_1pct_px) * 100.0)
    success_2pct_corner = float(np.mean(errors_px <= thresh_2pct_px) * 100.0)

    per_corner_means = np.mean(errors_px, axis=0)  # (4,) [TL, TR, BR, BL]

    metrics = {
        "mean_corner_error_px": mean_err_px,
        "mean_corner_error_pct": mean_err_pct,
        "success_rate_1pct": success_1pct_sample,
        "success_rate_2pct": success_2pct_sample,
        "success_rate_1pct_corner": success_1pct_corner,
        "success_rate_2pct_corner": success_2pct_corner,
        "TL_err_px": float(per_corner_means[0]),
        "TR_err_px": float(per_corner_means[1]),
        "BR_err_px": float(per_corner_means[2]),
        "BL_err_px": float(per_corner_means[3]),
        "diagonal_px": diag_px,
        "num_samples": N,
    }

    return metrics
