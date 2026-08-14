"""CLI Document Scanner Inference Script.

Fulfills Phase 08 CLI Inference requirements.
Usage:
python scan_document.py \
    --image data/real_photos/raw/20260803_112334.jpg \
    --output-dir outputs/scans \
    --corner-ckpt runs/exp-10_corner_approach_b/checkpoints/best.pt \
    --enh-ckpt runs/exp-008_enh_l1msssim_sobel/checkpoints/best.pt
"""

import argparse
import logging
from pathlib import Path
import sys
import cv2
import numpy as np

from src.pipeline.scanner import EndToEndScannerPipeline, DEFAULT_CORNER_CKPT, DEFAULT_ENH_CKPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_rgb_image(img_rgb: np.ndarray, output_path: Path) -> None:
    """Save an RGB numpy array to disk as a BGR image file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_img)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Document Scanner CLI: Corner Detection -> Homography Warp -> Document Enhancement"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/real_photos/raw/20260803_112334.jpg",
        help="Path to input raw smartphone photo",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/scans",
        help="Directory to save output scan stages and final clean image",
    )
    parser.add_argument(
        "--corner-ckpt",
        type=str,
        default=DEFAULT_CORNER_CKPT,
        help="Path to trained corner detection network checkpoint (.pt)",
    )
    parser.add_argument(
        "--enh-ckpt",
        type=str,
        default=DEFAULT_ENH_CKPT,
        help="Path to trained document enhancement network checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference ('cpu', 'cuda', 'cuda:0')",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Input image file not found: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing End-to-End Scanner Pipeline...")
    pipeline = EndToEndScannerPipeline(
        corner_ckpt=args.corner_ckpt,
        enh_ckpt=args.enh_ckpt,
        device=args.device,
    )

    logger.info(f"Processing input image: {image_path}")
    results = pipeline.scan(image_path)

    # Save stage images
    orig_file = output_dir / "01_original.png"
    overlay_file = output_dir / "02_corner_overlay.png"
    rectified_file = output_dir / "03_rectified.png"
    enhanced_file = output_dir / "04_enhanced_scan.png"

    save_rgb_image(results["original"], orig_file)
    save_rgb_image(results["corner_overlay"], overlay_file)
    save_rgb_image(results["rectified"], rectified_file)
    save_rgb_image(results["enhanced"], enhanced_file)

    logger.info("Pipeline execution complete! Saved outputs:")
    logger.info(f"  - Original photo:       {orig_file}")
    logger.info(f"  - Corner Overlay:        {overlay_file}")
    logger.info(f"  - Rectified Crop:        {rectified_file}")
    logger.info(f"  - Enhanced Clean Scan:   {enhanced_file}")

    corners_px = results["corners_px"]
    confidences = results["confidences"]
    logger.info("Predicted Corners (TL, TR, BR, BL):")
    labels = ["TL", "TR", "BR", "BL"]
    for i in range(4):
        logger.info(f"  {labels[i]}: ({corners_px[i, 0]:.1f}, {corners_px[i, 1]:.1f}) px [conf: {confidences[i]*100:.1f}%]")


if __name__ == "__main__":
    main()
