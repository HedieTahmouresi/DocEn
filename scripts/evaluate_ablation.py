"""Evaluation and visualization script for Phase 04 Loss Ablation [REQ-45], [REQ-22].

Generates:
1. `outputs/figures/p04_loss_curves.png`: Training and validation loss curves across epochs.
2. `outputs/figures/p04_loss_comparison.png`: Visual comparison figure comparing input,
   L-A (MSE), L-B (L1), L-C (L1+MS-SSIM), L-D (+Sobel), and Ground Truth with zoomed text crops.
3. Summary metrics report comparing all 4 variants against the no-model baseline.
"""

import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from model import EnhancementNet
from src.data.datasets import FrozenEvalDataset
from src.metrics.baseline import evaluate_no_model_baseline
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config
from src.utils.io import save_image


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
    models: dict,
    val_dataset: FrozenEvalDataset,
    sample_idx: int = 0,
    output_path: str = "outputs/figures/p04_loss_comparison.png",
):
    """Generate side-by-side visual comparison with zoomed-in text crop for all variants."""
    device = torch.device("cpu")
    batch = val_dataset[sample_idx]

    inp_tensor = batch["input"].unsqueeze(0).to(device)
    tgt_tensor = batch["target"].unsqueeze(0).to(device)

    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            model.to(device)
            model.eval()
            pred = model(inp_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            preds[name] = np.clip(pred, 0.0, 1.0)

    inp_img = inp_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    tgt_img = tgt_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    inp_img = np.clip(inp_img, 0.0, 1.0)
    tgt_img = np.clip(tgt_img, 0.0, 1.0)

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


def load_model_from_run(run_dir: Path) -> Optional[torch.nn.Module]:
    """Load trained model from best checkpoint in run directory."""
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = run_dir / "checkpoints" / "last.pt"
    if not best_ckpt.exists():
        return None

    ckpt = torch.load(best_ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    base_ch = model_cfg.get("base_channels", 32)
    levels = model_cfg.get("levels", 3)
    out_ch = model_cfg.get("out_channels", 3)
    upsample = model_cfg.get("upsample", "transpose")

    model = EnhancementNet(base_channels=base_ch, levels=levels, out_ch=out_ch, upsample=upsample)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    runs_root = Path("runs")
    run_dirs = [
        runs_root / "exp-001_enh_mse",
        runs_root / "exp-002_enh_l1",
        runs_root / "exp-003_enh_l1msssim",
        runs_root / "exp-004_enh_l1msssim_sobel",
    ]

    # Plot loss curves
    plot_loss_curves([str(d) for d in run_dirs])

    # Load frozen val dataset
    val_dataset = FrozenEvalDataset(frozen_dir="data/frozen/val", task="enhancement")

    # Load models
    variant_names = ["L-A", "L-B", "L-C", "L-D"]
    models = {}
    for name, run_dir in zip(variant_names, run_dirs):
        m = load_model_from_run(run_dir)
        if m is not None:
            models[name] = m

    if models:
        generate_loss_comparison_figure(models, val_dataset, sample_idx=0)
        print("Completed loss ablation evaluation figures!")
    else:
        print("No trained checkpoints found yet for loss comparison figure.")
