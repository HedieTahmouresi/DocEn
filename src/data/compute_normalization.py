"""
Compute per-channel normalization statistics from training split generated inputs only.

Fulfills REQ-13, ADR-009.
Prevents data leakage by ensuring validation and test splits are never included in normalization statistics.
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from src.utils.config import load_config
from src.data.datasets import SyntheticTrainDataset


def compute_dataset_normalization(
    scans_dir: Path,
    bg_dir: Path,
    splits_file: Path,
    generator_config: Dict[str, Any],
    num_samples: int = 2000,
    seed: int = 42
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute per-channel RGB mean and std over training split generated inputs.

    Args:
        scans_dir: Path to clean scans
        bg_dir: Path to background textures
        splits_file: Path to splits.json
        generator_config: Configuration dictionary for synthetic generator
        num_samples: Number of training samples to average over (default 2000)
        seed: Random seed for training dataset

    Returns:
        (mean, std) tuples for R, G, B channels in [0, 1]
    """
    print(f"Computing normalization statistics over {num_samples} training split samples...")

    dataset = SyntheticTrainDataset(
        scans_dir=scans_dir,
        bg_dir=bg_dir,
        splits_file=splits_file,
        split="train",
        task="enhancement",
        samples_per_epoch=num_samples,
        generator_config=generator_config,
        seed=seed,
        normalize=False
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for i in range(num_samples):
        sample = dataset[i]
        inp = sample["input"]  # (3, 512, 512) float32 in [0, 1]

        # Accumulate per-channel sums
        c, h, w = inp.shape
        num_px = h * w
        total_pixels += num_px

        channel_sum += inp.sum(dim=(1, 2)).double()
        channel_sum_sq += (inp ** 2).sum(dim=(1, 2)).double()

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")

    mean = channel_sum / total_pixels
    var = (channel_sum_sq / total_pixels) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    mean_tuple = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_tuple = (float(std[0]), float(std[1]), float(std[2]))

    print(f"Computed Normalization Statistics (RGB in [0, 1]):")
    print(f"  Mean: {mean_tuple}")
    print(f"  Std:  {std_tuple}")

    return mean_tuple, std_tuple


def update_config_normalization(base_yaml_path: Path, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
    """Update configs/base.yaml with computed normalization parameters."""
    with open(base_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["normalization"] = {
        "mean": [round(m, 4) for m in mean],
        "std": [round(s, 4) for s in std]
    }

    with open(base_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {base_yaml_path} with normalization parameters.")


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    base_yaml = root_dir / "configs" / "base.yaml"
    config = load_config(root_dir=root_dir)

    scans_dir = root_dir / config["clean_scans_dir"]
    bg_dir = root_dir / config["backgrounds_dir"]
    splits_file = root_dir / config["splits_file"]

    mean, std = compute_dataset_normalization(
        scans_dir=scans_dir,
        bg_dir=bg_dir,
        splits_file=splits_file,
        generator_config=config,
        num_samples=2000,
        seed=42
    )

    update_config_normalization(base_yaml, mean, std)


if __name__ == "__main__":
    main()
