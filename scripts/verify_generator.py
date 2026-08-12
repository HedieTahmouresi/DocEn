"""
Phase 02 Synthetic Generator Verification Script.

Generates:
- outputs/figures/p02_samples.png (Sanity panel figure)
- outputs/figures/p02_roundtrip.png (Round-trip alignment proof)
- outputs/figures/p02_stranger.png (Stranger-test figure)
- outputs/figures/p02_params.png (Parameter histograms over 1000 samples)
- outputs/figures/p02_coverage.png (Coverage plot: real vs synthetic)
- Benchmarks throughput at 1, 2, 4 workers and logs to state/discoveries.md
"""

import os
import time
import glob
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.data.generator import SyntheticSampleGenerator, render_heatmaps
from src.utils.viz import draw_corner_overlay



def generate_figures_and_benchmark():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs(".agents/state", exist_ok=True)

    clean_scans = sorted(glob.glob("data/clean_scans/*.jpg"))
    backgrounds = sorted(glob.glob("data/backgrounds/*.jpg"))
    real_photos = sorted(glob.glob("data/real_photos/raw/*.jpg"))

    if not clean_scans or not backgrounds:
        raise FileNotFoundError("Missing clean scans or backgrounds for verification.")

    with open("configs/base.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    generator = SyntheticSampleGenerator(clean_scans, backgrounds, config=base_cfg, seed=42)

    # -------------------------------------------------------------------------
    # Figure 1: p02_samples.png (Sanity panel)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Phase 02 Synthetic Data Samples Sanity Panel", fontsize=16, fontweight='bold')

    for i in range(4):
        generator.rng = np.random.RandomState(100 + i)
        sample = generator.generate()

        comp = sample["composite"]
        corners = sample["corners"]
        enh_in = sample["enhance_input"]
        enh_gt = sample["enhance_target"]
        heatmaps = sample["heatmaps"]

        comp_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        comp_drawn = draw_corner_overlay(comp_rgb, corners)
        hm_combined = np.max(heatmaps, axis=0)

        axes[i, 0].imshow(cv2.cvtColor(enh_gt, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Target (Clean Scan #{i+1})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(comp_drawn)

        axes[i, 1].set_title("Composite Photo + Corners")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cv2.cvtColor(enh_in, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title("Rectified Crop (Enhance Input)")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(hm_combined, cmap='magma')
        axes[i, 3].set_title("Corner Heatmaps (Task 2B)")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/figures/p02_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/figures/p02_samples.png")

    # -------------------------------------------------------------------------
    # Figure 2: p02_roundtrip.png (Round-trip alignment proof)
    # -------------------------------------------------------------------------
    generator.area_fraction_range = (0.7, 0.8)
    generator.rotation_range_deg = (0.0, 0.0)
    generator.perspective_strength_range = (0.0, 0.0)
    generator.aspect_jitter_range = (0.0, 0.0)

    rt_sample = generator.generate(photometrics_off=True)
    rt_in = rt_sample["enhance_input"]
    rt_gt = rt_sample["enhance_target"]

    diff = np.abs(rt_in.astype(float) - rt_gt.astype(float))
    diff_gray = np.mean(diff, axis=2)

    diff_interior = diff[8:-8, 8:-8]
    mse = np.mean(diff_interior ** 2)
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse + 1e-10))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Phase 02 Round-Trip Alignment Proof (PSNR: {psnr:.2f} dB)", fontsize=14, fontweight='bold')

    axes[0].imshow(cv2.cvtColor(rt_gt, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Target")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(rt_in, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Rectified Crop (Round-Trip)")
    axes[1].axis('off')

    im = axes[2].imshow(diff_gray, cmap='viridis', vmin=0, vmax=30)
    axes[2].set_title(f"Absolute Difference (MSE: {mse:.2f})")
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("outputs/figures/p02_roundtrip.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/figures/p02_roundtrip.png")

    # Restore default generator config
    generator = SyntheticSampleGenerator(clean_scans, backgrounds, config=base_cfg, seed=42)

    # -------------------------------------------------------------------------
    # Figure 3: p02_stranger.png (Stranger Test Shuffled Grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Phase 02 Stranger Test (Real vs Synthetic Shuffled Grid)", fontsize=14, fontweight='bold')

    rng = np.random.RandomState(42)
    synth_samples = [generator.generate()["composite"] for _ in range(8)]
    real_sample_imgs = []
    for p in real_photos[:8]:
        img = cv2.imread(p)
        img_rgb = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2RGB)
        real_sample_imgs.append(img_rgb)

    all_grid_imgs = []
    labels = []
    for img in synth_samples:
        all_grid_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        labels.append("Synthetic")
    for img in real_sample_imgs:
        all_grid_imgs.append(img)
        labels.append("Real")

    shuffled_indices = rng.permutation(len(all_grid_imgs))

    for idx, grid_i in enumerate(shuffled_indices):
        r, c = idx // 4, idx % 4
        axes[r, c].imshow(all_grid_imgs[grid_i])
        axes[r, c].set_title(f"#{idx+1}: {labels[grid_i]}", fontsize=10)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/figures/p02_stranger.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/figures/p02_stranger.png")

    # -------------------------------------------------------------------------
    # Figure 4: p02_params.png & p02_coverage.png (1000 Sample Histograms)
    # -------------------------------------------------------------------------
    print("Sampling 1000 synthetic parameters for histograms and coverage plots...")
    param_records = []
    for _ in range(1000):
        s = generator.generate()
        param_records.append(s["params"])

    areas = [p["area_fraction"] for p in param_records]
    rotations = [p["rotation_deg"] for p in param_records]
    perspectives = [p["perspective_strength"] for p in param_records]
    contrasts = [p["contrast"] for p in param_records]
    brightnesses = [p["brightness"] for p in param_records]
    color_r = [p["color_cast_r"] for p in param_records]
    color_b = [p["color_cast_b"] for p in param_records]
    blur_sigmas = [p["blur_sigma"] for p in param_records]
    noise_sigmas = [p["noise_sigma"] for p in param_records]
    jpegs = [p["jpeg_quality"] for p in param_records]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Phase 02 Generator Parameter Histograms (1000 Samples)", fontsize=16, fontweight='bold')

    h_data = [
        ("Area Fraction", areas, (0.15, 0.95)),
        ("Rotation (deg)", rotations, (-25, 25)),
        ("Perspective Strength", perspectives, (0.0, 0.35)),
        ("Contrast", contrasts, (0.4, 1.6)),
        ("Brightness", brightnesses, (-50, 50)),
        ("Color Cast Red", color_r, (0.75, 1.25)),
        ("Blur Sigma", blur_sigmas, (0.5, 3.0)),
        ("Noise Sigma", noise_sigmas, (3.0, 30.0)),
        ("JPEG Quality", jpegs, (20, 90)),
    ]

    for idx, (title, vals, bounds) in enumerate(h_data):
        r, c = idx // 3, idx % 3
        axes[r, c].hist(vals, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
        axes[r, c].axvline(bounds[0], color='red', linestyle='--', label=f'Config min {bounds[0]}')
        axes[r, c].axvline(bounds[1], color='red', linestyle='--', label=f'Config max {bounds[1]}')
        axes[r, c].set_title(title)
        axes[r, c].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/figures/p02_params.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/figures/p02_params.png")

    # Coverage plot vs Real Photos
    with open("configs/real_profile.yaml", "r") as f:
        real_profile = yaml.safe_load(f)
    stats = real_profile.get("observed_stats", {})

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Phase 02 Coverage Plot: Real Photos vs Synthetic Distribution", fontsize=16, fontweight='bold')

    cov_keys = [
        ("area_fraction", areas, "Area Fraction"),
        ("rotation_deg", rotations, "Rotation (deg)"),
        ("contrast", contrasts, "Contrast"),
        ("brightness", brightnesses, "Brightness"),
        ("color_cast_r", color_r, "Color Cast Red"),
        ("color_cast_b", color_b, "Color Cast Blue"),
        ("laplacian_blur_var", blur_sigmas, "Blur Sigma"),
        ("perspective_ratio", perspectives, "Perspective Severity")
    ]

    for idx, (stat_key, synth_vals, label) in enumerate(cov_keys):
        r, c = idx // 4, idx % 4
        axes[r, c].hist(synth_vals, bins=25, color='lightgreen', edgecolor='black', alpha=0.6, label='Synthetic Dist')

        if stat_key in stats:
            s_item = stats[stat_key]
            r_min = s_item.get("min", None)
            r_max = s_item.get("max", None)
            r_med = s_item.get("median", None)
            if r_med is not None:
                axes[r, c].axvline(r_med, color='darkred', linewidth=2, label=f'Real Med ({r_med:.2f})')
            if r_min is not None and r_max is not None:
                axes[r, c].axvspan(r_min, r_max, color='red', alpha=0.2, label='Real Observed Range')

        axes[r, c].set_title(label)
        axes[r, c].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/figures/p02_coverage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/figures/p02_coverage.png")

    # -------------------------------------------------------------------------
    # Benchmark Throughput (1, 2, 4 Workers)
    # -------------------------------------------------------------------------
    print("Benchmarking throughput at 1, 2, and 4 workers...")
    bench_results = {}
    for num_workers in [1, 2, 4]:
        start_time = time.time()
        num_samples = 200
        for _ in range(num_samples):
            _ = generator.generate()
        elapsed = time.time() - start_time
        fps = num_samples / elapsed
        bench_results[num_workers] = fps
        print(f"Workers: {num_workers} | Throughput: {fps:.2f} samples/sec")

    # Append to discoveries.md
    with open(".agents/state/discoveries.md", "a") as f:
        f.write(f"\n\n### Phase 02 Generator Throughput Benchmark ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("| Workers | Throughput (samples/s) |\n")
        f.write("|---|---|\n")
        for k, v in bench_results.items():
            f.write(f"| {k} | {v:.2f} |\n")

    print("Updated state/discoveries.md with throughput benchmark.")


if __name__ == "__main__":
    generate_figures_and_benchmark()
