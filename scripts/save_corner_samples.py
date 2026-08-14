"""Script to generate and save corner detection overlay samples on real smartphone photos.

Fulfills conventions §8 (TL Red, TR Green, BR Blue, BL Yellow) and saves visualization figures to:
outputs/figures/p06_corner_overlays/
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from src.data.datasets import RealPhotoDataset
from src.pipeline.corners import CornerPipeline, visualize_corner_overlay
from src.utils.config import load_config
from src.utils.io import save_image


def generate_corner_overlay_samples(
    ckpt_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    num_samples: int = 10,
) -> Path:
    """Generate corner overlay figures on real smartphone photos."""
    base_cfg = load_config(env="local_cpu")
    runs_dir = Path(base_cfg.get("runs_root", "runs"))

    if output_dir is None:
        output_dir = Path("outputs/figures/p06_corner_overlays")
    output_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_path is None or not ckpt_path.exists():
        # Find best checkpoint for Approach B (or Approach A fallback)
        b_candidates = sorted(list(runs_dir.glob("*corner_approach_b/checkpoints/best.pt")))
        a_candidates = sorted(list(runs_dir.glob("*corner_approach_a/checkpoints/best.pt")))

        if b_candidates:
            ckpt_path = b_candidates[0]
        elif a_candidates:
            ckpt_path = a_candidates[0]
        else:
            raise FileNotFoundError("No trained corner checkpoints found in runs/ directory!")

    print(f"Loading corner pipeline from: {ckpt_path}")
    pipeline = CornerPipeline(checkpoint_path=ckpt_path)

    raw_photos_dir = base_cfg["raw_photos_dir"]
    ref_scans_dir = base_cfg["reference_scans_dir"]
    ann_file = base_cfg["annotations_file"]

    if not (Path(raw_photos_dir).exists() and Path(ann_file).exists()):
        print(f"Real photos directory not found: {raw_photos_dir}")
        return output_dir

    real_dataset = RealPhotoDataset(
        raw_dir=raw_photos_dir,
        ref_dir=ref_scans_dir,
        ann_file=ann_file,
        task="corner",
        normalize=False,
    )

    limit = min(num_samples, len(real_dataset))
    print(f"Generating {limit} corner overlay samples...")

    for i in range(limit):
        item = real_dataset[i]
        img_rgb = item["input"].numpy().transpose(1, 2, 0)  # [H, W, 3] uint8
        gt_corners_norm = item["target_corners"].numpy()   # [4, 2] in [0, 1]

        orig_h, orig_w = img_rgb.shape[:2]
        gt_corners_px = gt_corners_norm.copy()
        gt_corners_px[:, 0] *= float(orig_w)
        gt_corners_px[:, 1] *= float(orig_h)

        # Predict corners with model
        pred_corners_px, confidences = pipeline.predict(img_rgb)

        # Render predicted overlay
        overlay = visualize_corner_overlay(
            img_rgb,
            pred_corners_px,
            confidences=confidences,
            line_thickness=3,
            circle_radius=7,
        )

        # Render ground-truth hollow circles (white/black outline) for visual comparison
        for j in range(4):
            gx, gy = int(round(gt_corners_px[j, 0])), int(round(gt_corners_px[j, 1]))
            cv2.circle(overlay, (gx, gy), 11, (255, 255, 255), 2)
            cv2.circle(overlay, (gx, gy), 9, (0, 0, 0), 1)

        out_path = output_dir / f"sample_{i+1:02d}_{item['photo_id']}.png"
        save_image(out_path, overlay)
        print(f"  Saved sample overlay: {out_path}")

    print(f"\nAll corner overlay samples saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    generate_corner_overlay_samples()
