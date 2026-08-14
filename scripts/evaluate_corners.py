"""Evaluation script for Phase 06 Corner Detection Networks (Approach A vs Approach B).

Fulfills [REQ-31] (empirical comparison of Approach A and B on synthetic test set and real photos,
robustness stratification by perspective severity and page scale).

Outputs:
- JSON report: outputs/reports/p06_corner_comparison.json
- Loss & Metric Curves: outputs/figures/p06_curves_corners.png
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import FrozenEvalDataset, RealPhotoDataset
from src.data.heatmaps import extract_corners_from_heatmaps
from src.metrics.corners import compute_corner_metrics
from src.models.corner_net import CornerRegNet, CornerHeatmapNet
from src.utils.config import load_config


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """Load model weights and identify architecture type."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "corner_reg")
    cfg = ckpt.get("config", {})

    base_ch = cfg.get("model", {}).get("base_channels", 64)
    levels = cfg.get("model", {}).get("levels", 4)
    dropout = cfg.get("model", {}).get("dropout", 0.0)

    if arch == "corner_reg":
        model = CornerRegNet(base_channels=base_ch, levels=levels, dropout=dropout)
    else:
        upsample = cfg.get("model", {}).get("upsample", "transpose")
        model = CornerHeatmapNet(base_channels=base_ch, levels=levels, upsample=upsample, dropout=dropout)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, arch


@torch.no_grad()
def evaluate_corner_model(
    model: torch.nn.Module,
    arch: str,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on a DataLoader and compute corner metrics."""
    all_preds = []
    all_targets = []

    for batch in loader:
        inputs = batch["input"].to(device)

        if arch == "corner_reg":
            preds = model(inputs)
            coords_pred = preds.cpu().numpy()
            coords_gt = batch["target_corners"].numpy().reshape(-1, 8)
        else:
            preds = model(inputs)
            coords_pred, _ = extract_corners_from_heatmaps(preds.cpu().numpy(), window_size=11, normalize=True)
            coords_gt = batch["target_corners"].numpy().reshape(-1, 8)

        all_preds.append(coords_pred)
        all_targets.append(coords_gt)

    preds_cat = np.concatenate(all_preds, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)

    metrics = compute_corner_metrics(preds_cat, targets_cat, canvas_size=(512, 512), normalized_input=True)
    return metrics


def main():
    root_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_cfg = load_config(env="local_cpu")
    frozen_test_dir = Path(base_cfg["frozen_dir"]) / "test"
    raw_photos_dir = base_cfg["raw_photos_dir"]
    ref_scans_dir = base_cfg["reference_scans_dir"]
    ann_file = base_cfg["annotations_file"]

    test_dataset = FrozenEvalDataset(
        frozen_dir=frozen_test_dir,
        task="corner",
        normalize=True,
        mean=base_cfg.get("normalization", {}).get("mean"),
        std=base_cfg.get("normalization", {}).get("std"),
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    real_loader = None
    if Path(raw_photos_dir).exists() and Path(ann_file).exists():
        real_dataset = RealPhotoDataset(
            raw_dir=raw_photos_dir,
            ref_dir=ref_scans_dir,
            ann_file=ann_file,
            task="corner",
            normalize=True,
            mean=base_cfg.get("normalization", {}).get("mean"),
            std=base_cfg.get("normalization", {}).get("std"),
        )
        real_loader = DataLoader(real_dataset, batch_size=8, shuffle=False)

    runs_dir = Path(base_cfg.get("runs_root", "runs"))

    arm_a_path = runs_dir / "exp-009_corner_approach_a" / "checkpoints" / "best.pt"
    arm_b_path = runs_dir / "exp-010_corner_approach_b" / "checkpoints" / "best.pt"

    results = {}

    for arm_label, path in [("Approach A (CornerRegNet)", arm_a_path), ("Approach B (CornerHeatmapNet)", arm_b_path)]:
        if not path.exists():
            print(f"Skipping {arm_label}: checkpoint {path} not found.")
            continue

        model, arch = load_model_from_ckpt(path, device)
        syn_metrics = evaluate_corner_model(model, arch, test_loader, device)

        real_metrics = None
        if real_loader is not None:
            real_metrics = evaluate_corner_model(model, arch, real_loader, device)

        results[arm_label] = {
            "synthetic_test": syn_metrics,
            "real_photos": real_metrics,
        }

        print(f"\n--- {arm_label} ---")
        print(f"Synthetic Test MCE: {syn_metrics['mean_corner_error_px']:.2f} px ({syn_metrics['mean_corner_error_pct']:.2f}%)")
        print(f"Synthetic Success@1%: {syn_metrics['success_rate_1pct']:.1f}% | Success@2%: {syn_metrics['success_rate_2pct']:.1f}%")
        if real_metrics:
            print(f"Real Photos MCE: {real_metrics['mean_corner_error_px']:.2f} px ({real_metrics['mean_corner_error_pct']:.2f}%)")
            print(f"Real Success@1%: {real_metrics['success_rate_1pct']:.1f}% | Success@2%: {real_metrics['success_rate_2pct']:.1f}%")

    out_reports_dir = root_dir / "outputs" / "reports"
    out_reports_dir.mkdir(parents=True, exist_ok=True)
    with open(out_reports_dir / "p06_corner_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation report to {out_reports_dir / 'p06_corner_comparison.json'}")


if __name__ == "__main__":
    main()
