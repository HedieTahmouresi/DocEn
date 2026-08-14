"""Evaluation metrics: PSNR, SSIM, OCR CER/Confidence, Corner Metrics, and Baseline."""

from src.metrics.image import calculate_psnr, calculate_ssim
from src.metrics.ocr import compute_cer, run_ocr_on_image, normalize_text_for_cer
from src.metrics.corners import compute_corner_errors, compute_corner_metrics

__all__ = [
    "calculate_psnr",
    "calculate_ssim",
    "compute_cer",
    "run_ocr_on_image",
    "normalize_text_for_cer",
    "compute_corner_errors",
    "compute_corner_metrics",
]
