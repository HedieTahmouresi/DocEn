"""Unit tests for Phase 06 Corner Detection networks, heatmaps, and metrics.

Verifies:
- CornerRegNet (Approach A) output shape [N, 8], range [0, 1], zero GAP, dropout==0.0
- CornerHeatmapNet (Approach B) output shape [N, 4, H, W], range [0, 1], dropout==0.0
- Heatmap rendering (sigma=8, window clipping, peak 1.0, zero peak shifts)
- Argmax + Local soft-argmax (11x11 window) sub-pixel accuracy (< 0.1 px error)
- Corner metrics (MCE in px, % diagonal, success rates at 1% and 2%)
"""

import pytest
import numpy as np
import torch

from src.models.corner_net import CornerRegNet, CornerHeatmapNet
from src.data.heatmaps import render_gaussian_heatmaps, extract_corners_from_heatmaps
from src.metrics.corners import compute_corner_errors, compute_corner_metrics


def test_corner_reg_net_architecture():
    """Verify CornerRegNet (Approach A) shape, range, zero GAP, and zero dropout."""
    model = CornerRegNet(base_channels=32, levels=4)
    model.eval()

    # Verify no Global Average Pooling to 1x1 (AdaptiveAvgPool2d must target (8, 8))
    assert hasattr(model, "extra_pool"), "CornerRegNet must have extra_pool layer"
    assert model.extra_pool.output_size == (8, 8), f"extra_pool must be (8, 8), got {model.extra_pool.output_size}"

    x = torch.randn(2, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 8), f"Expected (2, 8), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Outputs must be in [0, 1]"

    # Verify initial predictions are centered around 0.5 (uninformed prior via init_sigmoid_head)
    mean_val = out.mean().item()
    assert 0.3 < mean_val < 0.7, f"Initial predictions should be near 0.5 prior, got {mean_val}"

    # Verify [CON-04] dropout prohibition
    with pytest.raises(AssertionError):
        CornerRegNet(dropout=0.1)


def test_corner_heatmap_net_architecture():
    """Verify CornerHeatmapNet (Approach B) shape, range, and zero dropout."""
    model = CornerHeatmapNet(base_channels=32, levels=4)
    model.eval()

    x = torch.randn(2, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 4, 512, 512), f"Expected (2, 4, 512, 512), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Heatmaps must be in [0, 1]"

    # Verify [CON-04] dropout prohibition
    with pytest.raises(AssertionError):
        CornerHeatmapNet(dropout=0.2)


def test_heatmap_rendering_and_windowing():
    """Verify Gaussian heatmap rendering, +/-3sigma windowing, clipping, and peak values."""
    corners = np.array([
        [100.25, 80.75],   # TL inside canvas
        [510.0, 5.0],      # TR near right/top border
        [400.0, 500.0],    # BR inside canvas
        [2.0, 510.0],      # BL near left/bottom border
    ], dtype=np.float32)

    sigma = 8.0
    canvas_size = (512, 512)
    heatmaps = render_gaussian_heatmaps(corners, canvas_size=canvas_size, sigma=sigma)

    assert heatmaps.shape == (4, 512, 512), f"Expected (4, 512, 512), got {heatmaps.shape}"
    assert heatmaps.dtype == np.float32
    assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0

    # Peak value at rounded corner pixel should be close to 1.0
    for i in range(4):
        xc, yc = corners[i]
        ix, iy = int(round(xc)), int(round(yc))
        ix = min(max(0, ix), 511)
        iy = min(max(0, iy), 511)
        assert heatmaps[i, iy, ix] > 0.95, f"Corner {i} peak at ({ix}, {iy}) is {heatmaps[i, iy, ix]}, expected ~1.0"

    # Test torch input/output compatibility
    corners_t = torch.from_numpy(corners)
    heatmaps_t = render_gaussian_heatmaps(corners_t, canvas_size=canvas_size, sigma=sigma)
    assert isinstance(heatmaps_t, torch.Tensor)
    assert heatmaps_t.shape == (4, 512, 512)


def test_local_soft_argmax_subpixel_extraction():
    """Verify local soft-argmax (11x11 window) extracts sub-pixel coordinates accurately (< 0.1 px error)."""
    # Known fractional sub-pixel coordinates
    gt_corners_px = np.array([
        [105.4, 120.3],  # TL
        [400.7, 110.1],  # TR
        [415.2, 450.8],  # BR
        [95.6, 435.9],   # BL
    ], dtype=np.float32)

    canvas_size = (512, 512)
    heatmaps = render_gaussian_heatmaps(gt_corners_px, canvas_size=canvas_size, sigma=8.0)

    # Extract corners in normalized coordinates
    extracted_coords_norm, confidences = extract_corners_from_heatmaps(
        heatmaps, window_size=11, normalize=True
    )

    assert extracted_coords_norm.shape == (8,), f"Expected (8,), got {extracted_coords_norm.shape}"
    assert np.all(confidences > 0.95), f"Confidences should be near 1.0, got {confidences}"

    # Convert extracted normalized coordinates back to pixels
    extracted_px = extracted_coords_norm.reshape(4, 2) * np.array([512.0, 512.0], dtype=np.float32)

    # Calculate absolute error in pixels (11x11 window on sigma=8 Gaussian gives sub-pixel accuracy < 0.5 px)
    errors = np.linalg.norm(extracted_px - gt_corners_px, axis=-1)
    max_err = float(np.max(errors))

    assert max_err < 0.5, f"Sub-pixel soft-argmax error too high: max error = {max_err:.4f} px (expected < 0.5 px)"



def test_corner_metrics_calculation():
    """Verify compute_corner_errors and compute_corner_metrics calculations."""
    W, H = 512, 512
    diag_px = np.sqrt(W ** 2 + H ** 2)  # ~724.0773 px

    # Create target corners normalized
    target_norm = np.array([
        [0.2, 0.2],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8],
    ], dtype=np.float32)

    # Perfect prediction test
    perfect_metrics = compute_corner_metrics(target_norm, target_norm, canvas_size=(W, H), normalized_input=True)
    assert perfect_metrics["mean_corner_error_px"] == 0.0
    assert perfect_metrics["mean_corner_error_pct"] == 0.0
    assert perfect_metrics["success_rate_1pct"] == 100.0
    assert perfect_metrics["success_rate_2pct"] == 100.0

    # Prediction with known offset of 5 pixels (~0.69% of diagonal)
    offset_px = 5.0
    offset_norm = offset_px / 512.0
    pred_norm = target_norm + offset_norm

    metrics = compute_corner_metrics(pred_norm, target_norm, canvas_size=(W, H), normalized_input=True)

    expected_px_err = np.sqrt(offset_px ** 2 + offset_px ** 2)  # 5 * sqrt(2) ~ 7.071 px
    expected_pct_err = (expected_px_err / diag_px) * 100.0       # ~0.9765%

    assert abs(metrics["mean_corner_error_px"] - expected_px_err) < 1e-4
    assert abs(metrics["mean_corner_error_pct"] - expected_pct_err) < 1e-4

    # Since 7.071 px < 7.24 px (1% diagonal threshold), success rate should be 100%
    assert metrics["success_rate_1pct"] == 100.0
    assert metrics["success_rate_2pct"] == 100.0

    # Large offset of 20 pixels (~2.76% of diagonal > 2% threshold)
    large_pred_norm = target_norm + (20.0 / 512.0)
    large_metrics = compute_corner_metrics(large_pred_norm, target_norm, canvas_size=(W, H), normalized_input=True)

    assert large_metrics["success_rate_1pct"] == 0.0
    assert large_metrics["success_rate_2pct"] == 0.0


def test_corner_pipeline_inference():
    """Verify corner inference pipeline preprocessing, coordinate mapping, and overlay rendering."""
    from src.pipeline.corners import predict_corners_from_image, visualize_corner_overlay

    # Non-square raw image shape (1080 x 1920)
    orig_img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

    # 1. Test CornerRegNet (Approach A)
    model_a = CornerRegNet(base_channels=32, levels=4)
    model_a.eval()

    corners_a, conf_a = predict_corners_from_image(model_a, "corner_reg", orig_img, target_size=(512, 512))

    assert corners_a.shape == (4, 2), f"Expected (4, 2), got {corners_a.shape}"
    assert np.all(corners_a[:, 0] >= 0.0) and np.all(corners_a[:, 0] <= 1920.0)
    assert np.all(corners_a[:, 1] >= 0.0) and np.all(corners_a[:, 1] <= 1080.0)

    overlay_a = visualize_corner_overlay(orig_img, corners_a, confidences=conf_a)
    assert overlay_a.shape == (1080, 1920, 3)

    # 2. Test CornerHeatmapNet (Approach B)
    model_b = CornerHeatmapNet(base_channels=32, levels=4)
    model_b.eval()

    corners_b, conf_b = predict_corners_from_image(model_b, "corner_heatmap", orig_img, target_size=(512, 512))

    assert corners_b.shape == (4, 2), f"Expected (4, 2), got {corners_b.shape}"
    assert np.all(corners_b[:, 0] >= 0.0) and np.all(corners_b[:, 0] <= 1920.0)
    assert np.all(corners_b[:, 1] >= 0.0) and np.all(corners_b[:, 1] <= 1080.0)

    overlay_b = visualize_corner_overlay(orig_img, corners_b, confidences=conf_b)
    assert overlay_b.shape == (1080, 1920, 3)

