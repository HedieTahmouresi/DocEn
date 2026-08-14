"""Script to generate and save side-by-side corner detection comparison samples for Approach A and Approach B.

Fulfills [REQ-31] & conventions §8 (TL Red, TR Green, BR Blue, BL Yellow).
Saves side-by-side comparison figures to:
outputs/figures/p06_corner_overlays/
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

from src.data.datasets import RealPhotoDataset
from src.pipeline.corners import CornerPipeline, visualize_corner_overlay
from src.utils.config import load_config
from src.utils.io import save_image


def add_title_header(img_rgb: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Add a clean top banner with title text to an image."""
    h, w = img_rgb.shape[:2]
    header_h = 60
    banner = np.zeros((header_h, w, 3), dtype=np.uint8) + 30  # Dark grey background

    cv2.putText(
        banner,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            banner,
            subtitle,
            (15, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([banner, img_rgb])


def generate_corner_overlay_samples(
    output_dir: Optional[Path] = None,
    num_samples: int = 10,
) -> Path:
    """Generate side-by-side corner overlay comparison figures for Approach A and Approach B."""
    base_cfg = load_config(env="local_cpu")
    runs_dir = Path(base_cfg.get("runs_root", "runs"))

    if output_dir is None:
        output_dir = Path("outputs/figures/p06_corner_overlays")
    output_dir.mkdir(parents=True, exist_ok=True)

    a_candidates = sorted(list(runs_dir.glob("*corner_approach_a/checkpoints/inference_best.pt")) + list(runs_dir.glob("*corner_approach_a/checkpoints/best.pt")))
    b_candidates = sorted(list(runs_dir.glob("*corner_approach_b/checkpoints/inference_best.pt")) + list(runs_dir.glob("*corner_approach_b/checkpoints/best.pt")))


    pipeline_a = None
    if a_candidates:
        print(f"Loading Approach A pipeline from: {a_candidates[0]}")
        pipeline_a = CornerPipeline(checkpoint_path=a_candidates[0])
    else:
        print("Approach A checkpoint not found; rendering Approach B only.")

    pipeline_b = None
    if b_candidates:
        print(f"Loading Approach B pipeline from: {b_candidates[0]}")
        pipeline_b = CornerPipeline(checkpoint_path=b_candidates[0])
    else:
        print("Approach B checkpoint not found; rendering Approach A only.")

    if pipeline_a is None and pipeline_b is None:
        raise FileNotFoundError("No trained corner checkpoints found in runs/ directory!")

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
    print(f"Generating {limit} side-by-side corner overlay comparison figures...")

    for i in range(limit):
        item = real_dataset[i]
        sample_name = item.get("name", item.get("photo_id", f"{i+1:02d}.jpg"))
        photo_id = Path(sample_name).stem

        inp = item["input"].numpy().transpose(1, 2, 0)
        if inp.dtype != np.uint8:
            img_rgb = (np.clip(inp, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_rgb = inp

        gt_corners_norm = item["target_corners"].numpy()   # [4, 2] in [0, 1]
        orig_h, orig_w = img_rgb.shape[:2]

        gt_corners_px = gt_corners_norm.copy()
        gt_corners_px[:, 0] *= float(orig_w)
        gt_corners_px[:, 1] *= float(orig_h)

        panels = []

        # Panel A: Approach A (Coordinate Regression)
        if pipeline_a is not None:
            pred_a, conf_a = pipeline_a.predict(img_rgb)
            overlay_a = visualize_corner_overlay(img_rgb, pred_a, confidences=conf_a, line_thickness=3, circle_radius=7)
            for j in range(4):
                gx, gy = int(round(gt_corners_px[j, 0])), int(round(gt_corners_px[j, 1]))
                cv2.circle(overlay_a, (gx, gy), 11, (255, 255, 255), 2)
                cv2.circle(overlay_a, (gx, gy), 9, (0, 0, 0), 1)
            panel_a = add_title_header(overlay_a, "Approach A: Coordinate Regression", "Stuck at center prior (MCE ~232 px)")
            panels.append(panel_a)

        # Panel B: Approach B (Heatmap Regression)
        if pipeline_b is not None:
            pred_b, conf_b = pipeline_b.predict(img_rgb)
            overlay_b = visualize_corner_overlay(img_rgb, pred_b, confidences=conf_b, line_thickness=3, circle_radius=7)
            for j in range(4):
                gx, gy = int(round(gt_corners_px[j, 0])), int(round(gt_corners_px[j, 1]))
                cv2.circle(overlay_b, (gx, gy), 11, (255, 255, 255), 2)
                cv2.circle(overlay_b, (gx, gy), 9, (0, 0, 0), 1)
            panel_b = add_title_header(overlay_b, "Approach B: Heatmap Regression", "Sub-pixel U-Net heatmaps (MCE ~62 px)")
            panels.append(panel_b)

        if len(panels) == 2:
            comparison_img = np.hstack(panels)
        else:
            comparison_img = panels[0]

        out_path = output_dir / f"sample_{i+1:02d}_{photo_id}_comparison.png"
        save_image(comparison_img, out_path)
        print(f"  Saved side-by-side comparison: {out_path.name}")

    print(f"\nAll side-by-side corner comparison samples saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    generate_corner_overlay_samples()
