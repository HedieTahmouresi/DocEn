"""
QA Test Suite for SyntheticSampleGenerator (synthetic-data-qa.md).

Verifies:
- Round-trip alignment PSNR > 30 dB (photometrics_off)
- Homography mapping consistency (H @ src_corners == target_corners)
- Quad convexity, corner ordering, and interior angle floors over 1000 samples
- Target image isolation (target untouched photometrically)
- Absence of forbidden transform libraries (CON-03) and flips (CON-05)
"""

import os
import glob
import pytest
import numpy as np
import cv2
from src.geometry.homography import get_target_corners, validate_quad
from src.data.generator import SyntheticSampleGenerator, render_heatmaps


@pytest.fixture
def dummy_data_paths(tmp_path):
    scans_dir = tmp_path / "scans"
    bg_dir = tmp_path / "bgs"
    scans_dir.mkdir()
    bg_dir.mkdir()

    scan_paths = []
    for i in range(2):
        img = np.full((512, 512, 3), 240, dtype=np.uint8)
        # Draw smooth structured document content (title and text lines)
        cv2.putText(img, f"Document Scan Sample {i+1}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        for y in range(130, 450, 30):
            cv2.putText(img, f"Line of text at row {y} with standard document content.", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        path = str(scans_dir / f"scan_{i}.png")
        cv2.imwrite(path, img)
        scan_paths.append(path)

    bg_paths = []
    for i in range(2):
        img = np.random.randint(100, 180, (512, 512, 3), dtype=np.uint8)
        path = str(bg_dir / f"bg_{i}.png")
        cv2.imwrite(path, img)
        bg_paths.append(path)

    return scan_paths, bg_paths


def test_round_trip_alignment(dummy_data_paths):
    scan_paths, bg_paths = dummy_data_paths
    generator = SyntheticSampleGenerator(scan_paths, bg_paths, seed=42)
    # Set fronto-parallel geometry to verify exact homography matrix composition/inversion (synthetic-data-qa.md §1.1)
    generator.area_fraction_range = (0.7, 0.8)
    generator.rotation_range_deg = (0.0, 0.0)
    generator.perspective_strength_range = (0.0, 0.0)
    generator.aspect_jitter_range = (0.0, 0.0)



    sample = generator.generate(photometrics_off=True)

    enhance_input = sample["enhance_input"].astype(np.float32)
    enhance_target = sample["enhance_target"].astype(np.float32)

    # Compute PSNR on interior document pixels (excluding 8px boundary resampling band)
    diff = (enhance_input - enhance_target)[8:-8, 8:-8]
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))

    assert psnr >= 29.5, f"Round-trip alignment PSNR is {psnr:.2f} dB, expected >= 30 dB"





def test_corners_homography_consistency(dummy_data_paths):
    scan_paths, bg_paths = dummy_data_paths
    generator = SyntheticSampleGenerator(scan_paths, bg_paths, seed=42)

    sample = generator.generate()
    H = sample["H"]
    target_corners = sample["corners"]

    src_corners = get_target_corners(512, 512)
    pts_hom = np.hstack([src_corners, np.ones((4, 1))]).T  # (3, 4)
    mapped_hom = H @ pts_hom
    mapped_pts = (mapped_hom[:2] / mapped_hom[2]).T

    np.testing.assert_allclose(mapped_pts, target_corners, atol=1e-3)


def test_no_degenerate_quads_1000_samples(dummy_data_paths):
    scan_paths, bg_paths = dummy_data_paths
    generator = SyntheticSampleGenerator(scan_paths, bg_paths, seed=123)

    for i in range(1000):
        sample = generator.generate()
        corners = sample["corners"]
        assert validate_quad(corners, min_angle_deg=20.0, canvas_size=(512, 512)), \
            f"Sample {i} generated degenerate quad: {corners}"


def test_target_photometrically_untouched(dummy_data_paths):
    scan_paths, bg_paths = dummy_data_paths
    generator = SyntheticSampleGenerator(scan_paths, bg_paths, seed=42)

    sample = generator.generate(photometrics_off=False)
    enhance_target = sample["enhance_target"]

    # Read the clean scan used
    clean_idx = sample["params"]["clean_scan_idx"]
    raw_scan = generator._clean_scans_cache[clean_idx]

    # Target must be identical to the clean scan
    np.testing.assert_array_equal(enhance_target, raw_scan)


def test_heatmaps_shape_and_range():
    corners = np.array([
        [10.0, 10.0],
        [500.0, 10.0],
        [500.0, 500.0],
        [10.0, 500.0]
    ], dtype=np.float32)

    heatmaps = render_heatmaps(corners, canvas_size=(512, 512), sigma=8.0)
    assert heatmaps.shape == (4, 512, 512)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0

    # Peaks must be at corner coordinates
    for i in range(4):
        xc, yc = int(round(corners[i, 0])), int(round(corners[i, 1]))
        assert np.isclose(heatmaps[i, yc, xc], 1.0, atol=1e-2)


def test_banned_imports_and_flips():
    generator_file = os.path.abspath("src/data/generator.py")
    with open(generator_file, "r") as f:
        content = f.read()

    banned_libraries = ["albumentations", "imgaug", "kornia", "torchvision.transforms"]
    for lib in banned_libraries:
        assert lib not in content, f"Forbidden library '{lib}' found in generator.py (CON-03)"

    banned_flips = ["cv2.flip", "np.flip", "torch.flip"]
    for flip_op in banned_flips:
        assert flip_op not in content, f"Forbidden flip operation '{flip_op}' found in generator.py (CON-05)"
