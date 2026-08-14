"""Unit tests for standalone document enhancement pipeline and edge case handling."""

import os
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch

from src.models.unet import EnhancementNet
from src.pipeline.enhance import (
    preprocess_image_for_enhancement,
    postprocess_enhancement_output,
    enhance_document,
)


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a lightweight temporary checkpoint file for testing."""
    model = EnhancementNet(in_ch=3, base_channels=16, levels=3, out_ch=3)
    ckpt_path = tmp_path / "dummy_best.pt"
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model": {"base_channels": 16, "levels": 3, "upsample": "transpose"}
        },
    }
    torch.save(ckpt, ckpt_path)
    return str(ckpt_path)


def test_preprocessing_odd_aspect_ratio():
    # Odd aspect ratio image: 719 x 1033
    raw_img = np.random.randint(0, 256, (1033, 719, 3), dtype=np.uint8)
    tensor, orig_dims = preprocess_image_for_enhancement(raw_img, target_size=(512, 512))

    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == torch.float32
    assert orig_dims == (719, 1033)


def test_preprocessing_greyscale_jpeg():
    # Single-channel greyscale image: 600 x 800
    grey_img = np.random.randint(0, 256, (600, 800), dtype=np.uint8)
    tensor, orig_dims = preprocess_image_for_enhancement(grey_img, target_size=(512, 512))

    assert tensor.shape == (1, 3, 512, 512)
    assert orig_dims == (800, 600)


def test_postprocessing_dimensions_and_types():
    pred_tensor = torch.rand(1, 3, 512, 512)
    orig_dims = (1234, 567)  # (W_orig, H_orig)

    enhanced_np, enhanced_pil = postprocess_enhancement_output(pred_tensor, orig_dims)

    assert enhanced_np.shape == (567, 1234, 3)
    assert enhanced_np.dtype == np.uint8
    assert enhanced_pil.size == (1234, 567)


def test_full_pipeline_with_checkpoint(dummy_checkpoint):
    # Test full end-to-end pipeline run
    raw_img = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
    enhanced_np, enhanced_pil = enhance_document(
        raw_img, checkpoint_path=dummy_checkpoint, device="cpu"
    )

    assert enhanced_np.shape == (400, 300, 3)
    assert enhanced_np.dtype == np.uint8
    assert enhanced_pil.size == (300, 400)
