"""Unit tests for End-to-End Scanner Pipeline and Warp interface [REQ-40]."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.geometry.warp import warp_perspective
from src.pipeline.scanner import EndToEndScannerPipeline, DEFAULT_CORNER_CKPT, DEFAULT_ENH_CKPT


def test_warp_perspective_shape():
    """Verify warp_perspective returns specified target size."""
    img_rgb = np.zeros((300, 400, 3), dtype=np.uint8)
    img_rgb[50:250, 50:350] = 255  # White rectangle

    corners = np.array([
        [50.0, 50.0],
        [350.0, 50.0],
        [350.0, 250.0],
        [50.0, 250.0]
    ], dtype=np.float32)

    rectified = warp_perspective(img_rgb, corners, target_size=(512, 512))
    assert isinstance(rectified, np.ndarray)
    assert rectified.shape == (512, 512, 3)
    assert rectified.dtype == np.uint8


def test_end_to_end_scanner_pipeline_dummy_image():
    """Test EndToEndScannerPipeline end-to-end execution on a synthetic image."""
    corner_ckpt = Path(DEFAULT_CORNER_CKPT)
    enh_ckpt = Path(DEFAULT_ENH_CKPT)

    if not (corner_ckpt.exists() and enh_ckpt.exists()):
        pytest.skip(f"Checkpoints not available: {corner_ckpt}, {enh_ckpt}")

    pipeline = EndToEndScannerPipeline(
        corner_ckpt=corner_ckpt,
        enh_ckpt=enh_ckpt,
        device="cpu",
    )

    # Create dummy 600x800 RGB image
    dummy_img = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)

    results = pipeline.scan(dummy_img)

    assert isinstance(results, dict)
    assert "original" in results
    assert "corner_overlay" in results
    assert "rectified" in results
    assert "enhanced" in results
    assert "corners_px" in results
    assert "confidences" in results

    assert results["original"].shape == (600, 800, 3)
    assert results["corner_overlay"].shape == (600, 800, 3)
    assert results["rectified"].shape == (512, 512, 3)
    assert results["corners_px"].shape == (4, 2)
    assert results["confidences"].shape == (4,)
    assert isinstance(results["enhanced"], np.ndarray)


def test_end_to_end_scanner_robustness_formats():
    """Test pipeline robustness across greyscale, RGBA, and non-square images."""
    corner_ckpt = Path(DEFAULT_CORNER_CKPT)
    enh_ckpt = Path(DEFAULT_ENH_CKPT)

    if not (corner_ckpt.exists() and enh_ckpt.exists()):
        pytest.skip("Checkpoints not available")

    pipeline = EndToEndScannerPipeline(
        corner_ckpt=corner_ckpt,
        enh_ckpt=enh_ckpt,
        device="cpu",
    )

    # 1. Greyscale image (400, 300)
    grey_img = np.full((400, 300), 128, dtype=np.uint8)
    res_grey = pipeline.scan(grey_img)
    assert res_grey["original"].shape == (400, 300, 3)
    assert res_grey["rectified"].shape == (512, 512, 3)

    # 2. RGBA image (300, 500, 4)
    rgba_img = np.full((300, 500, 4), 200, dtype=np.uint8)
    res_rgba = pipeline.scan(rgba_img)
    assert res_rgba["original"].shape == (300, 500, 3)
    assert res_rgba["rectified"].shape == (512, 512, 3)

    # 3. Extreme non-square aspect ratio (150, 900, 3)
    aspect_img = np.full((150, 900, 3), 180, dtype=np.uint8)
    res_aspect = pipeline.scan(aspect_img)
    assert res_aspect["original"].shape == (150, 900, 3)
    assert res_aspect["rectified"].shape == (512, 512, 3)
