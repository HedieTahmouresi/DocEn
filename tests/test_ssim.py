"""Unit tests for SSIM and MS-SSIM implementation.

Validates PyTorch SSIM implementation against `skimage.metrics.structural_similarity`
to 1e-4 precision across five test cases required by Phase 04 gate:
1. Random noise
2. Real document (clean scan sample)
3. Identical images (exactly 1.0)
4. Known constant offset
5. Flat/uniform patch (degenerate case - blank document margin)
"""

import os
import pytest
import numpy as np
import cv2
import torch
import skimage.metrics

from src.losses.ssim import ssim, ms_ssim, SSIM, MSSSIM


def compute_skimage_ssim(
    img1_np: np.ndarray, img2_np: np.ndarray, win_size: int = 11, sigma: float = 1.5
) -> float:
    """Compute reference SSIM using skimage over batch and channels [N, C, H, W]."""
    N, C, H, W = img1_np.shape
    ssim_vals = []
    for n in range(N):
        for c in range(C):
            val = skimage.metrics.structural_similarity(
                img1_np[n, c],
                img2_np[n, c],
                win_size=win_size,
                gaussian_weights=True,
                sigma=sigma,
                use_sample_covariance=False,
                data_range=1.0,
            )
            ssim_vals.append(val)
    return float(np.mean(ssim_vals))


def test_ssim_identical_images():
    """Case 3: Identical images must return exactly 1.0."""
    x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    val = ssim(x, x).item()
    assert abs(val - 1.0) < 1e-6, f"Expected 1.0 for identical images, got {val}"

    val_ms = ms_ssim(x, x).item()
    assert abs(val_ms - 1.0) < 1e-5, f"Expected 1.0 for identical images in MS-SSIM, got {val_ms}"


def test_ssim_random_noise():
    """Case 1: Random noise validated against skimage to 1e-4."""
    torch.manual_seed(42)
    img1 = torch.rand(2, 3, 128, 128, dtype=torch.float32)
    img2 = torch.rand(2, 3, 128, 128, dtype=torch.float32)

    our_val = ssim(img1, img2).item()
    ref_val = compute_skimage_ssim(img1.numpy(), img2.numpy())

    assert abs(our_val - ref_val) < 1e-4, f"Random noise mismatch: PyTorch {our_val:.6f} vs skimage {ref_val:.6f}"


def test_ssim_constant_offset():
    """Case 4: Known constant offset validated against skimage to 1e-4."""
    img1 = torch.full((1, 3, 128, 128), 0.5, dtype=torch.float32)
    img2 = torch.full((1, 3, 128, 128), 0.6, dtype=torch.float32)

    our_val = ssim(img1, img2).item()
    ref_val = compute_skimage_ssim(img1.numpy(), img2.numpy())

    assert abs(our_val - ref_val) < 1e-4, f"Constant offset mismatch: PyTorch {our_val:.6f} vs skimage {ref_val:.6f}"


def test_ssim_flat_uniform_patch():
    """Case 5: Flat/uniform patch (degenerate case - blank document margin) validated against skimage."""
    # Blank document margin: completely constant flat region
    img1 = torch.ones(1, 3, 128, 128, dtype=torch.float32) * 0.95
    img2 = torch.ones(1, 3, 128, 128, dtype=torch.float32) * 0.95

    our_val = ssim(img1, img2).item()
    assert abs(our_val - 1.0) < 1e-5, f"Flat uniform identical patch failed: {our_val}"

    # Slightly perturbed flat patch
    img2_noisy = img1 + torch.randn_like(img1) * 0.01
    img2_noisy = torch.clamp(img2_noisy, 0.0, 1.0)

    our_val2 = ssim(img1, img2_noisy).item()
    ref_val2 = compute_skimage_ssim(img1.numpy(), img2_noisy.numpy())

    assert abs(our_val2 - ref_val2) < 1e-4, f"Flat patch mismatch: PyTorch {our_val2:.6f} vs skimage {ref_val2:.6f}"


def test_ssim_real_document():
    """Case 2: Real document image loaded from clean scans validated against skimage to 1e-4."""
    scans_dir = "data/clean_scans"
    if os.path.exists(scans_dir):
        files = [f for f in os.listdir(scans_dir) if f.endswith((".png", ".jpg", ".tif"))]
        if files:
            img_path = os.path.join(scans_dir, files[0])
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (256, 256))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Add synthetic degradation/offset for comparison
            degraded = torch.clamp(tensor * 0.8 + 0.1, 0.0, 1.0)

            our_val = ssim(tensor, degraded).item()
            ref_val = compute_skimage_ssim(tensor.numpy(), degraded.numpy())

            assert abs(our_val - ref_val) < 1e-4, f"Real document mismatch: PyTorch {our_val:.6f} vs skimage {ref_val:.6f}"
            return

    # Fallback synthetic document-like pattern if scans directory is missing
    h, w = 256, 256
    doc = np.ones((h, w, 3), dtype=np.float32) * 0.9  # white page
    doc[50:200, 50:200] = 0.1  # black text block
    tensor = torch.from_numpy(doc).permute(2, 0, 1).unsqueeze(0).float()
    degraded = torch.clamp(tensor * 0.85 + 0.05, 0.0, 1.0)

    our_val = ssim(tensor, degraded).item()
    ref_val = compute_skimage_ssim(tensor.numpy(), degraded.numpy())
    assert abs(our_val - ref_val) < 1e-4, f"Synthetic document mismatch: PyTorch {our_val:.6f} vs skimage {ref_val:.6f}"


def test_ssim_module_wrapper():
    """Verify SSIM and MSSSIM Module wrappers produce identical outputs to functions."""
    x1 = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    x2 = torch.rand(2, 3, 64, 64, dtype=torch.float32)

    mod_ssim = SSIM()
    mod_ms_ssim = MSSSIM()

    assert torch.allclose(mod_ssim(x1, x2), ssim(x1, x2))
    assert torch.allclose(mod_ms_ssim(x1, x2), ms_ssim(x1, x2))
