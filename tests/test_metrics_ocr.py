"""Unit tests for image and OCR metric implementations."""

import pytest
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.metrics.image import calculate_psnr, calculate_ssim
from src.metrics.ocr import (
    normalize_text_for_cer,
    levenshtein_distance,
    compute_cer,
    run_ocr_on_image,
)


def test_psnr_and_ssim_sanity():
    # 1. Identical tensors -> PSNR capped at 100 dB, SSIM == 1.0
    img1 = torch.rand(2, 3, 64, 64)
    psnr_self = calculate_psnr(img1, img1)
    ssim_self = calculate_ssim(img1, img1)

    assert psnr_self == 100.0
    assert abs(ssim_self - 1.0) < 1e-4

    # 2. Completely inverted tensors -> PSNR near low value, SSIM low
    img2 = 1.0 - img1
    psnr_inv = calculate_psnr(img1, img2)
    ssim_inv = calculate_ssim(img1, img2)

    assert psnr_inv < 10.0
    assert ssim_inv < 0.5


def test_ocr_text_normalization():
    raw_text = "  Hello \n\n world! \t Spec   test. "
    normalized = normalize_text_for_cer(raw_text)

    # Must collapse spaces & newlines, strip ends, keep case & punctuation
    assert normalized == "Hello world! Spec test."


def test_levenshtein_distance():
    assert levenshtein_distance("cat", "hat") == 1
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("same", "same") == 0


def test_cer_calculation():
    gt = "The quick brown fox"
    pred_exact = "The quick brown fox"
    pred_one_err = "The quik brown fox"

    assert compute_cer(pred_exact, gt) == 0.0
    assert abs(compute_cer(pred_one_err, gt) - (1.0 / len(gt))) < 1e-4


def test_run_ocr_on_synthetic_image():
    # Create a simple synthetic image containing clear black text on white background
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), "DOCUMENT SCANNER", fill=(0, 0, 0))

    img_np = np.array(img)
    text, conf = run_ocr_on_image(img_np, psm=6)

    assert "DOCUMENT" in text.upper() or "SCANNER" in text.upper()
    assert conf > 0.0
