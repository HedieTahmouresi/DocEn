"""
Freezing script for validation and test sets.

Fulfills REQ-15, CON-07, ADR-003.
Generates frozen validation and test sets on disk as PNG images with JSON corner targets and manifests.
"""

import os
import json
import cv2
import datetime
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union

from src.utils.config import load_config
from src.data.generator import SyntheticSampleGenerator


def get_git_commit_hash() -> str:
    """Return short git commit hash or 'unknown' if git command fails."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def freeze_dataset(
    split: str,
    output_dir: Union[str, Path],
    scans_dir: Union[str, Path],
    bg_dir: Union[str, Path],
    splits_file: Union[str, Path],
    samples_per_scan: int,
    seed: int = 123,
    generator_config: Optional[Dict[str, Any]] = None,
    frozen_version: str = "v1.0"
) -> Dict[str, Any]:
    """
    Generate and freeze synthetic evaluation dataset to disk as lossless PNGs.

    Args:
        split: 'val' or 'test'
        output_dir: Target directory path for frozen dataset
        scans_dir: Path to clean scans directory
        bg_dir: Path to background textures directory
        splits_file: Path to splits.json
        samples_per_scan: Number of degradations to generate per source scan
        seed: Random seed for frozen set generation
        generator_config: Generator configuration dict
        frozen_version: Frozen contract version string

    Returns:
        Manifest dictionary written to manifest.json
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scans_dir = Path(scans_dir)
    bg_dir = Path(bg_dir)
    splits_file = Path(splits_file)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    if split not in splits_data:
        raise KeyError(f"Split '{split}' not found in {splits_file}")

    source_scans = sorted(splits_data[split])
    clean_scan_paths = [str(scans_dir / name) for name in source_scans]
    background_paths = sorted(
        [str(bg_dir / f) for f in os.listdir(bg_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    )

    print(f"Freezing split '{split}': {len(clean_scan_paths)} scans x {samples_per_scan} degradations = {len(clean_scan_paths) * samples_per_scan} samples.")

    generator = SyntheticSampleGenerator(
        clean_scan_paths=clean_scan_paths,
        background_paths=background_paths,
        config=generator_config,
        seed=seed
    )

    corners_manifest: Dict[str, list] = {}
    sample_counter = 0

    for scan_idx, scan_path in enumerate(clean_scan_paths):
        for deg_i in range(samples_per_scan):
            sample_id = f"{sample_counter:05d}"
            # Deterministic background selection
            bg_idx = (scan_idx * samples_per_scan + deg_i) % len(background_paths)

            sample = generator.generate(
                clean_scan_idx=scan_idx,
                background_idx=bg_idx
            )

            composite = sample["composite"]          # uint8 BGR
            enh_input = sample["enhance_input"]      # uint8 BGR
            enh_target = sample["enhance_target"]    # uint8 BGR
            corners = sample["corners"].tolist()      # [[x, y] x 4] absolute px @ 512

            # Write lossless PNGs (never JPEG per REQ-15 / ADR-003)
            comp_file = images_dir / f"{sample_id}_composite.png"
            enh_in_file = images_dir / f"{sample_id}_enh_input.png"
            enh_tgt_file = images_dir / f"{sample_id}_enh_target.png"

            success_comp = cv2.imwrite(str(comp_file), composite, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            success_in = cv2.imwrite(str(enh_in_file), enh_input, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            success_tgt = cv2.imwrite(str(enh_tgt_file), enh_target, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            if not (success_comp and success_in and success_tgt):
                raise RuntimeError(f"Failed to write PNG images for sample {sample_id}")

            corners_manifest[sample_id] = corners
            sample_counter += 1

    # Save corners.json
    corners_json_path = output_dir / "corners.json"
    with open(corners_json_path, "w", encoding="utf-8") as f:
        json.dump(corners_manifest, f, indent=2)

    # Save manifest.json
    manifest = {
        "frozen_version": frozen_version,
        "split": split,
        "sample_count": sample_counter,
        "seed": seed,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "source_scans": source_scans,
        "generator_config": generator_config or {}
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully froze {sample_counter} samples to {output_dir}")
    return manifest


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    config = load_config(root_dir=root_dir)

    scans_dir = root_dir / config["clean_scans_dir"]
    bg_dir = root_dir / config["backgrounds_dir"]
    splits_file = root_dir / config["splits_file"]
    frozen_base = root_dir / config["frozen_dir"]
    seed = config.get("frozen_seed", 123)

    # Val split: 5 scans x 100 samples/scan = 500 samples
    freeze_dataset(
        split="val",
        output_dir=frozen_base / "val",
        scans_dir=scans_dir,
        bg_dir=bg_dir,
        splits_file=splits_file,
        samples_per_scan=100,
        seed=seed,
        generator_config=config
    )

    # Test split: 4 scans x 125 samples/scan = 500 samples
    freeze_dataset(
        split="test",
        output_dir=frozen_base / "test",
        scans_dir=scans_dir,
        bg_dir=bg_dir,
        splits_file=splits_file,
        samples_per_scan=125,
        seed=seed + 1,
        generator_config=config
    )


if __name__ == "__main__":
    main()
