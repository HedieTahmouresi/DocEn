"""
Unit test for RealPhotoDataset and CON-06 degradation pipeline isolation assertion.
"""

import pytest
import torch
from pathlib import Path
from src.data.datasets import RealPhotoDataset


def test_real_photo_dataset_enhancement():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "real_photos" / "raw"
    ref_dir = root_dir / "data" / "real_photos" / "reference_scans"
    ann_file = root_dir / "data" / "real_photos" / "annotations" / "CV Doc-Enhancement Real Test Set.v1i.coco" / "train" / "_annotations.coco.json"

    dataset = RealPhotoDataset(raw_dir, ref_dir, ann_file, target_size=(512, 512), task="enhancement")
    assert len(dataset) == 30

    sample = dataset[0]
    assert "input" in sample and "target" in sample
    assert sample["input"].shape == (3, 512, 512)
    assert sample["target"].shape == (3, 512, 512)
    assert 0.0 <= sample["input"].min() and sample["input"].max() <= 1.0
    assert 0.0 <= sample["target"].min() and sample["target"].max() <= 1.0

    # CON-06 test
    with pytest.raises(RuntimeError):
        dataset.apply_degradation()


def test_real_photo_dataset_corner():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "real_photos" / "raw"
    ref_dir = root_dir / "data" / "real_photos" / "reference_scans"
    ann_file = root_dir / "data" / "real_photos" / "annotations" / "CV Doc-Enhancement Real Test Set.v1i.coco" / "train" / "_annotations.coco.json"

    dataset = RealPhotoDataset(raw_dir, ref_dir, ann_file, target_size=(512, 512), task="corner")
    assert len(dataset) == 30

    sample = dataset[0]
    assert "input" in sample and "target_corners" in sample
    assert sample["input"].shape == (3, 512, 512)
    assert sample["target_corners"].shape == (4, 2)
    assert 0.0 <= sample["target_corners"].min() and sample["target_corners"].max() <= 1.0


if __name__ == "__main__":
    test_real_photo_dataset_enhancement()
    test_real_photo_dataset_corner()
    print("All RealPhotoDataset unit tests passed successfully!")
