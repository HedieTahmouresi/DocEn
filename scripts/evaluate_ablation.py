"""Evaluation and visualization script for Phase 04 Loss Ablation [REQ-45], [REQ-22].

Generates:
1. `outputs/figures/p04_loss_curves.png`: Training and validation loss curves across epochs.
2. `outputs/figures/p04_loss_comparison.png`: Visual comparison figure comparing input,
   L-A (MSE), L-B (L1), L-C (L1+MS-SSIM), L-D (+Sobel), and Ground Truth with zoomed text crops.
3. `outputs/figures/p04_ablation_summary.json`: PSNR/SSIM per variant on the frozen
   validation set, with the [REQ-26] no-model baseline as the first row.

Validation only — [CON-07] holds the synthetic test split back until Phase 05, and
ADR-006 requires the ablation winner to be selected on validation.
"""

import csv
import json
import os
from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from model import EnhancementNet
from src.data.datasets import BaselineDataset, FrozenEvalDataset
from src.data.normalization import resolve_from_checkpoint
from src.metrics.baseline import evaluate_no_model_baseline
from src.metrics.image import calculate_psnr, calculate_ssim


def plot_loss_curves(run_dirs: list, output_path: str = "outputs/figures/p04_loss_curves.png"):
    """Plot train and validation loss curves for all four ablation runs."""
    plt.figure(figsize=(12, 5), dpi=300)

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    colors = {"exp-001": "#1f77b4", "exp-002": "#ff7f0e", "exp-003": "#2ca02c", "exp-004": "#d62728"}
    labels = {
        "exp-001": "L-A (MSE)",
        "exp-002": "L-B (L1)",
        "exp-003": "L-C (L1 + MS-SSIM)",
        "exp-004": "L-D (+ Sobel)",
    }

    for run_dir in run_dirs:
        csv_file = Path(run_dir) / "metrics.csv"
        if not csv_file.exists():
            continue

        exp_id = Path(run_dir).name.split("_")[0]
        color = colors.get(exp_id, None)
        label = labels.get(exp_id, exp_id)

        epochs, train_losses, val_losses = [], [], []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))

        ax1.plot(epochs, train_losses, label=label, color=color, linewidth=1.8)
        ax2.plot(epochs, val_losses, label=label, color=color, linewidth=1.8)

    ax1.set_title("Training Loss vs Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Training Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.set_title("Validation Loss vs Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curves to {output_path}")


def generate_loss_comparison_figure(
    variants: dict,
    display_dataset: "BaselineDataset",
    sample_idx: int = 0,
    output_path: str = "outputs/figures/p04_loss_comparison.png",
):
    """Generate side-by-side visual comparison with zoomed-in text crop for all variants.

    `variants` maps a label to `(model, dataset)`, where the dataset was built with
    *that checkpoint's* training-time input convention — a standardised model must
    never be fed [0, 1] input, or the reverse.

    The displayed input and target come from `display_dataset`, which is always
    un-standardised. ADR-009's trap is that a standardised tensor rendered raw is a
    contrast-mangled image, not the degraded photo the reader expects to see.
    """
    device = torch.device("cpu")

    display = display_dataset[sample_idx]
    inp_img = np.clip(display["input"].numpy().transpose(1, 2, 0), 0.0, 1.0)
    tgt_img = np.clip(display["target"].numpy().transpose(1, 2, 0), 0.0, 1.0)

    preds = {}
    with torch.no_grad():
        for name, (model, dataset) in variants.items():
            model.to(device)
            model.eval()
            model_input = dataset[sample_idx]["input"].unsqueeze(0).to(device)
            pred = model(model_input).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            preds[name] = np.clip(pred, 0.0, 1.0)

    # Crop box for text zoom (center crop)
    h, w, _ = inp_img.shape
    cy, cx = h // 2, w // 2
    crop_size = min(128, min(h, w) // 2)
    y1, y2 = cy - crop_size // 2, cy + crop_size // 2
    x1, x2 = cx - crop_size // 2, cx + crop_size // 2

    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5), dpi=300)

    titles = [
        "Degraded Input",
        "L-A (MSE)",
        "L-B (L1)",
        "L-C (L1+MS-SSIM)",
        "L-D (+Sobel)",
        "Ground Truth",
    ]

    images = [
        inp_img,
        preds.get("L-A", inp_img),
        preds.get("L-B", inp_img),
        preds.get("L-C", inp_img),
        preds.get("L-D", inp_img),
        tgt_img,
    ]

    for col in range(6):
        img = images[col]
        # Full view (top row)
        axes[0, col].imshow(img)
        axes[0, col].set_title(titles[col], fontsize=10, fontweight="bold")
        # Draw red crop rectangle
        rect = plt.Rectangle((x1, y1), crop_size, crop_size, fill=False, edgecolor="red", linewidth=1.5)
        axes[0, col].add_patch(rect)
        axes[0, col].axis("off")

        # Zoomed view (bottom row)
        crop_img = img[y1:y2, x1:x2]
        axes[1, col].imshow(crop_img)
        axes[1, col].set_title(f"Zoomed ({titles[col]})", fontsize=8)
        axes[1, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss comparison figure to {output_path}")


def load_variant_from_run(run_dir: Path, frozen_dir: Path):
    """Load a trained model plus the eval dataset matching its training-time input.

    The architecture and the input convention both come out of the checkpoint's own
    resolved config, never out of the current configs/ tree — a checkpoint trained
    before ADR-009 standardisation was wired up must still be evaluated on [0, 1]
    input, or its numbers are meaningless.
    """
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        return None

    # weights_only defaults to True from torch 2.6; the checkpoint carries a config dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    # Defaults match ADR-005, not the 32/3 smoke-test size the previous fallback used —
    # a wrong fallback here builds the wrong network and then fails on load_state_dict.
    model = EnhancementNet(
        base_channels=model_cfg.get("base_channels", 64),
        levels=model_cfg.get("levels", 4),
        out_ch=model_cfg.get("out_channels", 3),
        upsample=model_cfg.get("upsample", "transpose"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    standardize, mean, std = resolve_from_checkpoint(cfg)
    dataset = FrozenEvalDataset(
        frozen_dir=frozen_dir, task="enhancement", normalize=standardize, mean=mean, std=std
    )
    return model, dataset


@torch.no_grad()
def summarise_variants(variants: dict, baseline: dict, batch_size: int = 8) -> list:
    """PSNR/SSIM per variant on the frozen validation set, against the no-model baseline.

    Validation only. [CON-07] holds the test split back until the final evaluation, and
    ADR-006 requires the ablation winner to be picked on validation.
    """
    rows = [
        {
            "variant": "No-model baseline (degraded input)",
            "val_psnr": baseline["baseline_psnr"],
            "val_ssim": baseline["baseline_ssim"],
        }
    ]

    # 500 samples x 4 models at 512x512 is half an hour on CPU and seconds on the T4.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, (model, dataset) in variants.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        psnr_total, ssim_total, n = 0.0, 0.0, 0
        model.to(device).eval()
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            b = inputs.size(0)
            psnr_total += calculate_psnr(outputs, targets) * b
            ssim_total += calculate_ssim(outputs, targets) * b
            n += b
        rows.append(
            {"variant": name, "val_psnr": round(psnr_total / n, 4), "val_ssim": round(ssim_total / n, 4)}
        )
        model.cpu()

    return rows


def write_summary(rows: list, output_path: Path) -> None:
    """Write the ablation summary table to JSON and echo it to stdout."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    width = max(len(r["variant"]) for r in rows)
    print("\n" + "=" * (width + 26))
    print(f"{'Variant'.ljust(width)}  {'Val PSNR':>10}  {'Val SSIM':>10}")
    print("-" * (width + 26))
    for r in rows:
        print(f"{r['variant'].ljust(width)}  {r['val_psnr']:>10.4f}  {r['val_ssim']:>10.4f}")
    print("=" * (width + 26))
    print(f"Saved summary to {output_path}\n")


if __name__ == "__main__":
    runs_root = Path("runs")
    frozen_val_dir = Path("data/frozen/val")

    variant_names = ["L-A", "L-B", "L-C", "L-D"]
    run_dirs = [
        runs_root / "exp-001_enh_mse",
        runs_root / "exp-002_enh_l1",
        runs_root / "exp-003_enh_l1msssim",
        runs_root / "exp-004_enh_l1msssim_sobel",
    ]

    plot_loss_curves([str(d) for d in run_dirs])

    variants = {}
    for name, run_dir in zip(variant_names, run_dirs):
        loaded = load_variant_from_run(run_dir, frozen_val_dir)
        if loaded is not None:
            variants[name] = loaded

    if not variants:
        print("No trained checkpoints found yet — loss curves only.")
        raise SystemExit(0)

    display_dataset = BaselineDataset(frozen_dir=frozen_val_dir)
    generate_loss_comparison_figure(variants, display_dataset, sample_idx=0)

    baseline = evaluate_no_model_baseline(split="val")
    write_summary(
        summarise_variants(variants, baseline),
        Path("outputs/figures/p04_ablation_summary.json"),
    )
    print("Completed loss ablation evaluation figures and summary table!")
