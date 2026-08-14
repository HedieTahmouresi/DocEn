"""Heatmap Generation and Sub-pixel Coordinate Extraction.

Fulfills [REQ-30] (heatmap generation and sub-pixel extraction)
and ADR-008 (σ=8 px @512, ±3σ clipped window rendering, peak=1.0, local 11x11 soft-argmax).
"""

from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


def render_gaussian_heatmaps(
    corners: Union[np.ndarray, torch.Tensor],
    canvas_size: Tuple[int, int] = (512, 512),
    sigma: float = 8.0,
    normalized: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Render 2D Gaussian heatmaps for corner coordinates in a clipped +/-3*sigma window (ADR-008).

    Peak value of Gaussian is exactly 1.0 at the true corner (xc, yc).
    Window is clipped at canvas boundaries — never shifted, as shifting corrupts the label peak location.

    Args:
        corners: Corner coordinates array or tensor. Shapes supported:
                 - (4, 2) or (8,) for a single image
                 - (N, 4, 2) or (N, 8) for a batch of images
        canvas_size: (w, h) canvas size in pixels (default: 512, 512)
        sigma: Gaussian standard deviation in pixels (default: 8.0)
        normalized: If True, input corners are in [0, 1] and will be un-normalized to canvas_size.
                    If False, corners are assumed to be already in pixel space.

    Returns:
        heatmaps: (4, H, W) or (N, 4, H, W) float32 array/tensor matching input type, with values in [0, 1].
    """
    is_torch = isinstance(corners, torch.Tensor)
    device = corners.device if is_torch else None

    if is_torch:
        corners_np = corners.detach().cpu().numpy()
    else:
        corners_np = np.asarray(corners)

    orig_shape = corners_np.shape
    if corners_np.ndim == 1:
        # (8,) -> (1, 4, 2)
        corners_np = corners_np.reshape(1, 4, 2)
    elif corners_np.ndim == 2:
        if corners_np.shape == (4, 2):
            corners_np = corners_np[np.newaxis, ...]
        elif corners_np.shape == (1, 8) or corners_np.shape[1] == 8:
            corners_np = corners_np.reshape(-1, 4, 2)
        else:
            raise ValueError(f"Invalid corners shape: {orig_shape}")
    elif corners_np.ndim == 3 and corners_np.shape[1:] == (4, 2):
        pass
    else:
        raise ValueError(f"Invalid corners shape: {orig_shape}")

    N = corners_np.shape[0]
    w, h = canvas_size

    # Un-normalize if coordinates are in [0, 1]
    corners_px = corners_np.copy().astype(np.float32)
    if normalized:
        corners_px[:, :, 0] *= float(w)
        corners_px[:, :, 1] *= float(h)

    radius = int(np.ceil(3.0 * sigma))
    heatmaps_np = np.zeros((N, 4, h, w), dtype=np.float32)

    for n in range(N):
        for i in range(4):
            xc, yc = corners_px[n, i]

            # Bounding box window around corner
            x_min = max(0, int(np.floor(xc - radius)))
            x_max = min(w, int(np.ceil(xc + radius + 1)))
            y_min = max(0, int(np.floor(yc - radius)))
            y_max = min(h, int(np.ceil(yc + radius + 1)))

            if x_min >= x_max or y_min >= y_max:
                continue

            # Evaluate 2D Gaussian over window
            xs = np.arange(x_min, x_max, dtype=np.float32)
            ys = np.arange(y_min, y_max, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)

            dist_sq = (grid_x - xc) ** 2 + (grid_y - yc) ** 2
            gauss = np.exp(-dist_sq / (2.0 * sigma ** 2))

            heatmaps_np[n, i, y_min:y_max, x_min:x_max] = gauss

    if orig_shape == (4, 2) or orig_shape == (8,):
        out_np = heatmaps_np[0]
    else:
        out_np = heatmaps_np

    if is_torch:
        return torch.from_numpy(out_np).to(device)
    return out_np


def extract_corners_from_heatmaps(
    heatmaps: Union[np.ndarray, torch.Tensor],
    window_size: int = 11,
    normalize: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Extract corner coordinates from 4-channel heatmaps via Argmax + Local Soft-Argmax (ADR-008).

    Two-stage extraction:
    1. Global argmax per channel -> integer peak location (y_max, x_max).
    2. Local soft-argmax inside a `window_size` x `window_size` window centered on argmax peak
       -> intensity-weighted centroid sub-pixel coordinate.
    Also extracts max activation per channel as corner confidence score.

    Args:
        heatmaps: (4, H, W) or (N, 4, H, W) float array or torch Tensor in [0, 1].
        window_size: Odd integer size of local soft-argmax window (default: 11).
        normalize: If True, return coordinates normalized to [0, 1] by canvas dimensions.
                   If False, return coordinates in absolute pixel values.

    Returns:
        coords: (N, 8) or (8,) float coordinates [x0, y0, x1, y1, x2, y2, x3, y3] (TL, TR, BR, BL).
        confidences: (N, 4) or (4,) float peak confidence activations in [0, 1].
    """
    is_torch = isinstance(heatmaps, torch.Tensor)
    device = heatmaps.device if is_torch else None

    if is_torch:
        hm = heatmaps.detach().cpu().numpy()
    else:
        hm = np.asarray(heatmaps, dtype=np.float32)

    is_single = (hm.ndim == 3)
    if is_single:
        # (4, H, W) -> (1, 4, H, W)
        hm = hm[np.newaxis, ...]

    if hm.ndim != 4 or hm.shape[1] != 4:
        raise ValueError(f"Expected heatmaps shape (4, H, W) or (N, 4, H, W), got {heatmaps.shape}")

    N, C, H, W = hm.shape
    radius = window_size // 2

    coords_np = np.zeros((N, 4, 2), dtype=np.float32)
    confidences_np = np.zeros((N, 4), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            channel_hm = hm[n, c]

            # 1. Integer argmax peak
            max_idx = np.argmax(channel_hm)
            y_max, x_max = np.unravel_index(max_idx, (H, W))
            confidences_np[n, c] = channel_hm[y_max, x_max]

            # 2. Local window bounds
            x1 = max(0, x_max - radius)
            x2 = min(W, x_max + radius + 1)
            y1 = max(0, y_max - radius)
            y2 = min(H, y_max + radius + 1)

            window = channel_hm[y1:y2, x1:x2]
            sum_w = np.sum(window)

            if sum_w < 1e-8:
                # Fallback to integer argmax if window intensity is negligible
                sub_x = float(x_max)
                sub_y = float(y_max)
            else:
                grid_x = np.arange(x1, x2, dtype=np.float32)
                grid_y = np.arange(y1, y2, dtype=np.float32)

                sub_x = np.sum(grid_x * np.sum(window, axis=0)) / sum_w
                sub_y = np.sum(grid_y * np.sum(window, axis=1)) / sum_w

            if normalize:
                sub_x /= float(W)
                sub_y /= float(H)

            coords_np[n, c, 0] = sub_x
            coords_np[n, c, 1] = sub_y

    # Reshape coords to [N, 8] vector format [x0, y0, x1, y1, x2, y2, x3, y3]
    coords_vec = coords_np.reshape(N, 8)

    if is_single:
        coords_out = coords_vec[0]
        conf_out = confidences_np[0]
    else:
        coords_out = coords_vec
        conf_out = confidences_np

    if is_torch:
        return torch.from_numpy(coords_out).to(device), torch.from_numpy(conf_out).to(device)
    return coords_out, conf_out
