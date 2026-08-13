"""Evaluates the no-model baseline on the frozen validation set [REQ-26].

Measures PSNR and SSIM of degraded input images directly against ground truth
clean targets without any enhancement model.
"""

import json
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.datasets import FrozenEvalDataset
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config


def evaluate_no_model_baseline(config_path: Optional[str] = None, batch_size: int = 16) -> dict:
    """Compute PSNR and SSIM baseline on frozen validation set."""
    cfg = load_config(base_file=config_path)

    frozen_val_dir = Path(cfg.get("data", {}).get("frozen_val_dir", "data/frozen/val"))
    if not frozen_val_dir.exists():
        frozen_val_dir = Path("data/frozen/val")

    if not frozen_val_dir.exists():
        raise FileNotFoundError(f"Frozen val directory not found at {frozen_val_dir}. Run freeze script first.")

    val_dataset = FrozenEvalDataset(frozen_dir=frozen_val_dir, task="enhancement")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    psnr_total = 0.0
    ssim_total = 0.0
    num_samples = 0

    print(f"Evaluating no-model baseline on frozen val set ({len(val_dataset)} samples)...")

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"]
            targets = batch["target"]

            b_size = inputs.size(0)

            # Compute PSNR and SSIM between un-enhanced degraded input and clean target
            batch_psnr = calculate_psnr(inputs, targets)
            batch_ssim = calculate_ssim(inputs, targets)

            psnr_total += batch_psnr * b_size
            ssim_total += batch_ssim * b_size
            num_samples += b_size

    mean_psnr = psnr_total / num_samples
    mean_ssim = ssim_total / num_samples

    results = {
        "baseline_psnr": round(mean_psnr, 4),
        "baseline_ssim": round(mean_ssim, 4),
        "num_samples": num_samples,
    }

    print(f"\n=======================================================")
    print(f"No-Model Baseline -> PSNR: {results['baseline_psnr']:.4f} dB | SSIM: {results['baseline_ssim']:.4f}")
    print(f"=======================================================\n")
    return results


if __name__ == "__main__":
    evaluate_no_model_baseline()
