"""
Compute real photo calibration statistics across all active annotated real photos.

Measures:
- Page area fraction
- In-plane rotation (degrees)
- Perspective severity ratio
- Margin from frame edge
- Brightness and contrast within page
- Blur (Laplacian variance)
- Color cast (R/G, B/G ratios)

Writes configs/real_profile.yaml with observed distributions and widened generator ranges (1.5-2x).
"""

import os
import cv2
import yaml
import numpy as np
from pathlib import Path

from src.utils.io import load_image
from src.data.annotations import parse_coco_polygon_annotations


def compute_calibration():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "real_photos" / "raw"
    ann_file = root_dir / "data" / "real_photos" / "annotations" / "CV Doc-Enhancement Real Test Set.v1i.coco" / "train" / "_annotations.coco.json"
    out_yaml = root_dir / "configs" / "real_profile.yaml"

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
    parsed_anns = parse_coco_polygon_annotations(ann_file, active_filenames=raw_files)

    area_fractions = []
    rotations_deg = []
    perspective_ratios = []
    margins = []
    brightnesses = []
    contrasts = []
    blurs = []
    r_casts = []
    b_casts = []

    for name, corners in parsed_anns.items():
        img_path = raw_dir / name
        img_rgb = load_image(img_path)
        h, w, c = img_rgb.shape
        img_area = w * h
        diag = np.sqrt(w**2 + h**2)

        # 1. Page area fraction
        poly_pts = corners.astype(np.int32)
        poly_area = cv2.contourArea(poly_pts)
        area_fractions.append(poly_area / img_area)

        # 2. In-plane rotation (TL to TR angle)
        tl, tr, br, bl = corners
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        rotations_deg.append(abs(angle_deg))

        # 3. Perspective ratio (top edge vs bottom edge length)
        top_len = np.linalg.norm(tr - tl)
        bot_len = np.linalg.norm(br - bl)
        persp_ratio = max(top_len, bot_len) / (min(top_len, bot_len) + 1e-6)
        perspective_ratios.append(persp_ratio)

        # 4. Margin from frame edge (fraction of diag)
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])
        margin_px = min(min_x, w - max_x, min_y, h - max_y)
        margins.append(max(0, margin_px) / diag)

        # 5. Mask for internal page stats
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_pts], 255)
        page_pixels = img_rgb[mask == 255]

        if len(page_pixels) > 0:
            mean_rgb = np.mean(page_pixels, axis=0) # [R, G, B]
            brightness = np.mean(mean_rgb) / 255.0
            contrast = np.std(page_pixels) / 255.0
            brightnesses.append(brightness)
            contrasts.append(contrast)

            # Color cast relative to G
            g_val = max(1.0, mean_rgb[1])
            r_casts.append(mean_rgb[0] / g_val)
            b_casts.append(mean_rgb[2] / g_val)

        # Blur variance inside bounding box
        bx1, by1 = max(0, int(min_x)), max(0, int(min_y))
        bx2, by2 = min(w, int(max_x)), min(h, int(max_y))
        crop_gray = cv2.cvtColor(img_rgb[by1:by2, bx1:bx2], cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
        blurs.append(lap_var)

    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    profile = {
        "num_photos": len(parsed_anns),
        "observed_stats": {
            "area_fraction": stats(area_fractions),
            "rotation_deg": stats(rotations_deg),
            "perspective_ratio": stats(perspective_ratios),
            "margin_fraction": stats(margins),
            "brightness": stats(brightnesses),
            "contrast": stats(contrasts),
            "laplacian_blur_var": stats(blurs),
            "color_cast_r": stats(r_casts),
            "color_cast_b": stats(b_casts),
        },
        # Widened generator ranges (~1.5-2x spread per ADR-004 §3)
        "widened_generator_ranges": {
            "warp_range": [0.03, 0.35],          # Widened perspective warp displacement fraction
            "brightness_range": [0.35, 1.65],    # Widened brightness factor
            "contrast_range": [0.4, 1.6],        # Widened contrast factor
            "color_cast_r_range": [0.75, 1.25],  # Widened red cast
            "color_cast_b_range": [0.75, 1.25],  # Widened blue cast
            "downscale_range": [1.5, 4.5],       # Widened downscale resolution loss
            "blur_sigmas": [0.5, 3.0],           # Widened blur sigma
            "noise_sigmas": [3.0, 30.0],         # Widened noise sigma
            "jpeg_qualities": [20, 90],          # Widened JPEG compression quality range
        }
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(profile, f, default_flow_style=False)

    print(f"Computed calibration stats across {len(parsed_anns)} real photos.")
    print(f"Observed area fraction: {profile['observed_stats']['area_fraction']['p10']:.2f} - {profile['observed_stats']['area_fraction']['p90']:.2f}")
    print(f"Observed brightness: {profile['observed_stats']['brightness']['p10']:.2f} - {profile['observed_stats']['brightness']['p90']:.2f}")
    print(f"Wrote real-photo calibration profile to {out_yaml}")


if __name__ == "__main__":
    compute_calibration()
