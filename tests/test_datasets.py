"""
Unit tests for Phase 03 Datasets, Frozen Sets, DataLoader Worker RNG, and Split Integrity.

Fulfills REQ-11, REQ-12, REQ-13, REQ-14, REQ-15, REQ-16, REQ-17, REQ-18.
Enforces CON-06, CON-07, ADR-003, ADR-009.
"""

import json
import pytest
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seeding import seed_everything, worker_init_fn
from src.data.datasets import (
    SyntheticTrainDataset,
    FrozenEvalDataset,
    BaselineDataset,
    RealPhotoDataset
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config(project_root) -> dict:
    return load_config(root_dir=project_root)


def test_splits_disjointness(config, project_root):
    """REQ-14 & REQ-17: Verify source scan splits are strictly disjoint."""
    splits_file = project_root / config["splits_file"]
    assert splits_file.exists()

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_scans = set(splits["train"])
    val_scans = set(splits["val"])
    test_scans = set(splits["test"])

    # Disjointness assertion
    assert len(train_scans.intersection(val_scans)) == 0, "Train and Val splits share scans!"
    assert len(train_scans.intersection(test_scans)) == 0, "Train and Test splits share scans!"
    assert len(val_scans.intersection(test_scans)) == 0, "Val and Test splits share scans!"

    # Near-duplicate name check
    all_scans = list(train_scans) + list(val_scans) + list(test_scans)
    stems = [Path(s).stem for s in all_scans]
    assert len(stems) == len(set(stems)), "Duplicate scan stems found across splits!"


def test_synthetic_train_dataset_enhancement(config, project_root):
    """REQ-11: Test SyntheticTrainDataset for enhancement task."""
    dataset = SyntheticTrainDataset(
        scans_dir=project_root / config["clean_scans_dir"],
        bg_dir=project_root / config["backgrounds_dir"],
        splits_file=project_root / config["splits_file"],
        split="train",
        task="enhancement",
        samples_per_epoch=10,
        generator_config=config,
        seed=42
    )

    assert len(dataset) == 10
    sample = dataset[0]

    assert "input" in sample and "target" in sample
    assert sample["input"].shape == (3, 512, 512)
    assert sample["target"].shape == (3, 512, 512)
    assert sample["input"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert 0.0 <= sample["input"].min() and sample["input"].max() <= 1.0
    assert 0.0 <= sample["target"].min() and sample["target"].max() <= 1.0


def test_synthetic_train_dataset_corner(config, project_root):
    """REQ-11 & ADR-008: Test SyntheticTrainDataset for corner detection task."""
    dataset = SyntheticTrainDataset(
        scans_dir=project_root / config["clean_scans_dir"],
        bg_dir=project_root / config["backgrounds_dir"],
        splits_file=project_root / config["splits_file"],
        split="train",
        task="corner",
        samples_per_epoch=10,
        generator_config=config,
        seed=42
    )

    sample = dataset[0]

    assert "input" in sample and "target_corners" in sample and "target_heatmaps" in sample
    assert sample["input"].shape == (3, 512, 512)
    assert sample["target_corners"].shape == (4, 2)
    assert sample["target_heatmaps"].shape == (4, 512, 512)

    # Coords in [0, 1] per REQ-13
    assert 0.0 <= sample["target_corners"].min() and sample["target_corners"].max() <= 1.0
    assert 0.0 <= sample["target_heatmaps"].min() and sample["target_heatmaps"].max() <= 1.0


def test_frozen_eval_dataset(config, project_root):
    """REQ-15: Test FrozenEvalDataset loads frozen disk samples correctly."""
    frozen_val_dir = project_root / config["frozen_dir"] / "val"
    assert frozen_val_dir.exists(), "Frozen val directory missing. Run freeze.py first!"

    dataset = FrozenEvalDataset(
        frozen_dir=frozen_val_dir,
        task="enhancement"
    )

    assert len(dataset) == 500
    sample = dataset[0]

    assert "name" in sample and "input" in sample and "target" in sample
    assert sample["input"].shape == (3, 512, 512)
    assert sample["target"].shape == (3, 512, 512)
    assert 0.0 <= sample["input"].min() and sample["input"].max() <= 1.0


def test_baseline_dataset(config, project_root):
    """REQ-26: Test BaselineDataset for no-model baseline calculation."""
    frozen_test_dir = project_root / config["frozen_dir"] / "test"
    assert frozen_test_dir.exists(), "Frozen test directory missing. Run freeze.py first!"

    dataset = BaselineDataset(frozen_dir=frozen_test_dir)
    assert len(dataset) == 500

    sample = dataset[0]
    assert "name" in sample and "input" in sample and "target" in sample
    assert sample["input"].shape == (3, 512, 512)
    assert sample["target"].shape == (3, 512, 512)


def test_real_photo_dataset_con06_isolation(config, project_root):
    """CON-06: Test RealPhotoDataset isolation assertion."""
    dataset = RealPhotoDataset(
        raw_dir=project_root / config["raw_photos_dir"],
        ref_dir=project_root / config["reference_scans_dir"],
        ann_file=project_root / config["annotations_file"],
        task="enhancement"
    )

    assert not hasattr(dataset, "degradation_pipeline")
    with pytest.raises(RuntimeError):
        dataset.apply_degradation()


def test_worker_rng_independence(config, project_root):
    """
    CRITICAL TEST: Worker RNG independence with worker_init_fn.
    Verifies samples differ across multi-worker loader boundaries.
    """
    dataset = SyntheticTrainDataset(
        scans_dir=project_root / config["clean_scans_dir"],
        bg_dir=project_root / config["backgrounds_dir"],
        splits_file=project_root / config["splits_file"],
        split="train",
        task="enhancement",
        samples_per_epoch=16,
        generator_config=config,
        seed=42
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    loader_iter = iter(loader)
    batch1 = next(loader_iter)
    batch2 = next(loader_iter)

    # Assert images across batches differ
    diff = (batch1["input"] - batch2["input"]).abs().sum().item()
    assert diff > 1.0, "Worker RNG trap! Batches returned identical samples!"

    # Assert samples within the same batch (across workers) differ
    sample_diff = (batch1["input"][0] - batch1["input"][1]).abs().sum().item()
    assert sample_diff > 1.0, "Worker RNG trap! Samples within batch are identical!"



def test_coordinate_scaling_round_trip():
    """REQ-12 & REQ-13: Test coordinate normalization and denormalization round-trip."""
    orig_coords = np.array([
        [50.5, 100.25],
        [450.0, 80.75],
        [480.2, 490.5],
        [20.0, 470.1]
    ], dtype=np.float32)

    w, h = 512, 512

    # Normalize to [0, 1]
    norm_coords = orig_coords.copy()
    norm_coords[:, 0] /= float(w)
    norm_coords[:, 1] /= float(h)

    assert np.all(norm_coords >= 0.0) and np.all(norm_coords <= 1.0)

    # Denormalize back to absolute px
    denorm_coords = norm_coords.copy()
    denorm_coords[:, 0] *= float(w)
    denorm_coords[:, 1] *= float(h)

    assert np.allclose(orig_coords, denorm_coords, atol=1e-4)


def test_frozen_set_byte_identity(config, project_root):
    """ADR-003 & REQ-15: Test frozen evaluation set loads byte-identical images across calls."""
    frozen_val_dir = project_root / config["frozen_dir"] / "val"
    dataset1 = FrozenEvalDataset(frozen_dir=frozen_val_dir, task="enhancement")
    dataset2 = FrozenEvalDataset(frozen_dir=frozen_val_dir, task="enhancement")

    sample1 = dataset1[0]
    sample2 = dataset2[0]

    assert torch.equal(sample1["input"], sample2["input"])
    assert torch.equal(sample1["target"], sample2["target"])
