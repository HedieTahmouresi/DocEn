"""Evaluates the no-model "do nothing" baseline [REQ-26].

Measures PSNR and SSIM of the degraded input itself against the clean target, with no
enhancement model in the loop. Spec §3.3: "Compute it first. If your model's scores are
not clearly above this line, it is not earning its parameters."

The spec names the **test** bucket for the reported baseline row. The val split is
available here too, because the Phase 04 gate needs a comparable line to check the
ablation winner against without touching test ([CON-07]).
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data.datasets import BaselineDataset
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config


def evaluate_no_model_baseline(
    split: str = "val",
    config_path: Optional[str] = None,
    batch_size: int = 16,
    frozen_dir: Optional[Path] = None,
) -> dict:
    """Compute the PSNR/SSIM baseline on a frozen split.

    Args:
        split: "val" (Phase 04 monitoring) or "test" (the [REQ-26] reported row).
            Only pass "test" during the final evaluation — [CON-07].
    """
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    if frozen_dir is None:
        cfg = load_config(base_file=config_path)
        frozen_dir = Path(cfg.get("data_root", "data")) / "frozen" / split
    frozen_dir = Path(frozen_dir)

    if not frozen_dir.exists():
        raise FileNotFoundError(
            f"Frozen '{split}' directory not found at {frozen_dir}. Run `python -m src.data.freeze` first."
        )

    # BaselineDataset is FrozenEvalDataset with standardisation forced off: the baseline
    # compares the degraded input *as an image* against the target, so both sides have to
    # be in [0, 1] with data_range=1.0 (ADR-009).
    dataset = BaselineDataset(frozen_dir=frozen_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    psnr_total = 0.0
    ssim_total = 0.0
    num_samples = 0

    print(f"Evaluating no-model baseline on the frozen {split} set ({len(dataset)} samples)...")

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"]
            targets = batch["target"]
            b_size = inputs.size(0)

            psnr_total += calculate_psnr(inputs, targets) * b_size
            ssim_total += calculate_ssim(inputs, targets) * b_size
            num_samples += b_size

    results = {
        "split": split,
        "baseline_psnr": round(psnr_total / num_samples, 4),
        "baseline_ssim": round(ssim_total / num_samples, 4),
        "num_samples": num_samples,
    }

    print("=======================================================")
    print(
        f"No-Model Baseline ({split}) -> PSNR: {results['baseline_psnr']:.4f} dB "
        f"| SSIM: {results['baseline_ssim']:.4f}"
    )
    print("=======================================================\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-model baseline [REQ-26]")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate_no_model_baseline(split=args.split, batch_size=args.batch_size)
