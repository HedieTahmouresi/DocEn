"""
Verification script for real photo COCO annotations.

Visualizes all 30 active real photo annotations with the standard color code (conventions §8)
and asserts quad convexity, vertex count, and coordinate bounds.
Saves outputs/figures/p01_annotations.png.
"""

import math
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.io import load_image, save_image
from src.data.annotations import parse_coco_polygon_annotations, is_convex_quad
from src.utils.viz import draw_corner_overlay


def verify_all_annotations():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "real_photos" / "raw"
    ann_file = root_dir / "data" / "real_photos" / "annotations" / "CV Doc-Enhancement Real Test Set.v1i.coco" / "train" / "_annotations.coco.json"
    out_fig = root_dir / "outputs" / "figures" / "p01_annotations.png"

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
    print(f"Verifying annotations for {len(raw_files)} active real photos...")

    parsed = parse_coco_polygon_annotations(ann_file, active_filenames=raw_files)
    assert len(parsed) == len(raw_files), f"Expected {len(raw_files)} annotations, got {len(parsed)}"

    rendered_images = []
    names = []

    for name in raw_files:
        img_path = raw_dir / name
        img_rgb = load_image(img_path)
        h, w = img_rgb.shape[:2]

        corners = parsed[name]
        assert corners.shape == (4, 2), f"Corners shape mismatch for {name}: {corners.shape}"
        assert is_convex_quad(corners), f"Quad for {name} is non-convex!"

        # Coordinate bounds check
        assert np.all(corners[:, 0] >= -5) and np.all(corners[:, 0] <= w + 5), f"X coords out of bounds for {name}"
        assert np.all(corners[:, 1] >= -5) and np.all(corners[:, 1] <= h + 5), f"Y coords out of bounds for {name}"

        # Draw overlay
        overlay = draw_corner_overlay(img_rgb, corners, radius=25, thickness=5)
        rendered_images.append(overlay)
        names.append(name)

    # Grid visualization (e.g. 5x6 grid)
    n = len(rendered_images)
    cols = 5
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()

    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < n:
            ax.imshow(rendered_images[idx])
            ax.set_title(names[idx], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"All {len(parsed)} annotations successfully verified! Saved grid figure to {out_fig}")


if __name__ == "__main__":
    verify_all_annotations()
