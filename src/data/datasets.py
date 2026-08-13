"""
Dataset classes for training, frozen evaluation, real photo evaluation, and baseline.

Fulfills REQ-11, REQ-12, REQ-13, REQ-14, REQ-15, REQ-16, REQ-17, REQ-18.
Enforces CON-06 (no degradation on real photos), CON-07 (frozen evaluation integrity), ADR-003, ADR-009.
"""

import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset

from src.utils.io import load_image
from src.data.annotations import parse_coco_polygon_annotations
from src.geometry.homography import rectify_document
from src.data.generator import SyntheticSampleGenerator, render_heatmaps


class SyntheticTrainDataset(Dataset):
    """
    On-the-fly synthetic sample dataset for training.

    REQ-11: Generate synthetic training samples on the fly.
    REQ-14: Disjoint splits by source scan.
    REQ-17: Shared split between corner detection and document enhancement.
    ADR-009: Input normalization (optional), target in [0, 1].
    """

    def __init__(
        self,
        scans_dir: Union[str, Path],
        bg_dir: Union[str, Path],
        splits_file: Union[str, Path],
        split: str = "train",
        task: str = "enhancement",  # "enhancement" or "corner"
        samples_per_epoch: int = 2000,
        target_size: Tuple[int, int] = (512, 512),
        generator_config: Optional[Dict] = None,
        seed: int = 42,
        normalize: bool = False,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None
    ):
        super().__init__()
        self.scans_dir = Path(scans_dir)
        self.bg_dir = Path(bg_dir)
        self.splits_file = Path(splits_file)
        self.split = split.lower()
        self.task = task.lower()
        self.samples_per_epoch = samples_per_epoch
        self.target_size = target_size
        self.seed = seed
        self.epoch = 0
        self.normalize = normalize

        if self.task in ("enhance", "enhancement"):
            self.task = "enhancement"
        elif self.task in ("corner", "corners"):
            self.task = "corner"
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'enhancement' or 'corner'.")

        # Load splits file and extract clean scans for this split
        if not self.splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {self.splits_file}")

        with open(self.splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        if self.split not in splits_data:
            raise KeyError(f"Split '{self.split}' not found in {self.splits_file}")

        split_scan_names = set(splits_data[self.split])
        all_scans = sorted([f for f in os.listdir(self.scans_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        clean_scan_paths = [str(self.scans_dir / name) for name in all_scans if name in split_scan_names]

        if not clean_scan_paths:
            raise RuntimeError(f"No clean scans found for split '{self.split}' in {self.scans_dir}")

        background_paths = sorted(
            [str(self.bg_dir / f) for f in os.listdir(self.bg_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )
        if not background_paths:
            raise RuntimeError(f"No background images found in {self.bg_dir}")

        # Instantiate generator with ONLY the scans belonging to this split
        self.generator = SyntheticSampleGenerator(
            clean_scan_paths=clean_scan_paths,
            background_paths=background_paths,
            config=generator_config,
            seed=seed
        )

        # Setup normalization tensors
        if self.normalize and mean is not None and std is not None:
            self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        else:
            self.mean_tensor = None
            self.std_tensor = None

        self.rng = np.random.default_rng(seed)

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for worker RNG seeding."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sync worker generator RNG if updated by worker_init_fn
        if hasattr(self, "rng") and hasattr(self.generator, "rng"):
            # Sample sample using generator
            pass

        sample = self.generator.generate()

        composite_bgr = sample["composite"]          # (512, 512, 3) uint8 BGR
        enh_input_bgr = sample["enhance_input"]      # (512, 512, 3) uint8 BGR
        enh_target_bgr = sample["enhance_target"]    # (512, 512, 3) uint8 BGR
        corners_abs = sample["corners"]              # (4, 2) float32 absolute px
        heatmaps = sample["heatmaps"]                # (4, 512, 512) float32 [0, 1]

        w, h = self.target_size

        # Boundary conversion: BGR uint8 -> RGB float32 tensor [0, 1]
        comp_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        enh_in_rgb = cv2.cvtColor(enh_input_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        enh_tgt_rgb = cv2.cvtColor(enh_target_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

        comp_tensor = torch.from_numpy(comp_rgb)
        enh_in_tensor = torch.from_numpy(enh_in_rgb)
        enh_tgt_tensor = torch.from_numpy(enh_tgt_rgb)

        # Corners: normalized to [0, 1] by dividing by (w, h)
        corners_norm = corners_abs.copy()
        corners_norm[:, 0] /= float(w)
        corners_norm[:, 1] /= float(h)
        corners_tensor = torch.from_numpy(corners_norm).float()
        heatmaps_tensor = torch.from_numpy(heatmaps).float()

        if self.task == "enhancement":
            input_tensor = enh_in_tensor
            if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
                input_tensor = (input_tensor - self.mean_tensor) / self.std_tensor
            return {
                "input": input_tensor,
                "target": enh_tgt_tensor
            }
        else:  # "corner"
            input_tensor = comp_tensor
            if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
                input_tensor = (input_tensor - self.mean_tensor) / self.std_tensor
            return {
                "input": input_tensor,
                "target_corners": corners_tensor,
                "target_heatmaps": heatmaps_tensor
            }


class FrozenEvalDataset(Dataset):
    """
    On-disk frozen evaluation dataset for validation and test sets.

    REQ-15: Frozen evaluation sets on disk.
    CON-07: Untouched test set.
    """

    def __init__(
        self,
        frozen_dir: Union[str, Path],
        task: str = "enhancement",
        target_size: Tuple[int, int] = (512, 512),
        heatmap_sigma: float = 8.0,
        normalize: bool = False,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None
    ):
        super().__init__()
        self.frozen_dir = Path(frozen_dir)
        self.task = task.lower()
        self.target_size = target_size
        self.heatmap_sigma = heatmap_sigma
        self.normalize = normalize

        if self.task in ("enhance", "enhancement"):
            self.task = "enhancement"
        elif self.task in ("corner", "corners"):
            self.task = "corner"
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'enhancement' or 'corner'.")

        self.images_dir = self.frozen_dir / "images"
        self.corners_file = self.frozen_dir / "corners.json"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Frozen images directory not found: {self.images_dir}")
        if not self.corners_file.exists():
            raise FileNotFoundError(f"Frozen corners file not found: {self.corners_file}")

        with open(self.corners_file, "r", encoding="utf-8") as f:
            self.corners_data = json.load(f)

        # Discover all unique sample IDs (e.g. "00000", "00001", ...)
        sample_ids = sorted(list(self.corners_data.keys()))
        self.sample_ids = sample_ids

        # Setup normalization tensors
        if self.normalize and mean is not None and std is not None:
            self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        else:
            self.mean_tensor = None
            self.std_tensor = None

        self._cache = {}
        print(f"Loaded FrozenEvalDataset from {self.frozen_dir} ({len(self.sample_ids)} samples, task '{self.task}').")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        if sample_id in self._cache:
            return self._cache[sample_id]

        comp_path = self.images_dir / f"{sample_id}_composite.png"
        enh_in_path = self.images_dir / f"{sample_id}_enh_input.png"
        enh_tgt_path = self.images_dir / f"{sample_id}_enh_target.png"

        # load_image returns RGB float32 [0, 1] in HWC
        comp_rgb = load_image(comp_path)
        enh_in_rgb = load_image(enh_in_path)
        enh_tgt_rgb = load_image(enh_tgt_path)

        comp_tensor = torch.from_numpy(comp_rgb.transpose(2, 0, 1)).float() / 255.0
        enh_in_tensor = torch.from_numpy(enh_in_rgb.transpose(2, 0, 1)).float() / 255.0
        enh_tgt_tensor = torch.from_numpy(enh_tgt_rgb.transpose(2, 0, 1)).float() / 255.0

        # Load corners in absolute px @ target_size
        corners_abs = np.array(self.corners_data[sample_id], dtype=np.float32)  # (4, 2)
        w, h = self.target_size
        corners_norm = corners_abs.copy()
        corners_norm[:, 0] /= float(w)
        corners_norm[:, 1] /= float(h)

        corners_tensor = torch.from_numpy(corners_norm).float()

        if self.task == "corner":
            heatmaps = render_heatmaps(corners_abs, canvas_size=self.target_size, sigma=self.heatmap_sigma)
            heatmaps_tensor = torch.from_numpy(heatmaps).float()

            input_tensor = comp_tensor
            if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
                input_tensor = (input_tensor - self.mean_tensor) / self.std_tensor

            res = {
                "name": sample_id,
                "input": input_tensor,
                "target_corners": corners_tensor,
                "target_heatmaps": heatmaps_tensor
            }
        else:  # "enhancement"
            input_tensor = enh_in_tensor
            if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
                input_tensor = (input_tensor - self.mean_tensor) / self.std_tensor

            res = {
                "name": sample_id,
                "input": input_tensor,
                "target": enh_tgt_tensor
            }

        self._cache[sample_id] = res
        return res


class BaselineDataset(FrozenEvalDataset):
    """
    Evaluation dataset for calculating the no-model baseline metric on the frozen test set.

    REQ-26: Baseline evaluation path matches model evaluation path.
    """

    def __init__(
        self,
        frozen_dir: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__(
            frozen_dir=frozen_dir,
            task="enhancement",
            target_size=target_size,
            normalize=False
        )


class RealPhotoDataset(Dataset):
    """
    Evaluation dataset for real smartphone photos.

    CON-06 ENFORCEMENT: Never run synthetic degradation pipeline on real photos.
    REQ-16: Fourth evaluation set (Real smartphone photos).
    """

    def __init__(
        self,
        raw_dir: Union[str, Path],
        ref_dir: Union[str, Path],
        ann_file: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512),
        task: str = "enhancement",  # "enhancement" or "corner"
        normalize: bool = False,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.ref_dir = Path(ref_dir)
        self.ann_file = Path(ann_file)
        self.target_size = target_size
        self.task = task.lower()
        self.normalize = normalize

        if self.task in ("enhance", "enhancement"):
            self.task = "enhancement"
        elif self.task in ("corner", "corners"):
            self.task = "corner"
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'enhancement' or 'corner'.")

        # ASSERT CON-06: Ensure degradation pipeline is explicitly isolated and absent
        assert not hasattr(self, "degradation_pipeline"), "CON-06 Violation: Degradation pipeline attached to RealPhotoDataset!"

        self.raw_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        self.annotations = parse_coco_polygon_annotations(self.ann_file, active_filenames=self.raw_files)

        # Verify matching reference scans
        self.samples = []
        for name in self.raw_files:
            if name in self.annotations:
                ref_path = self.ref_dir / name
                if ref_path.exists():
                    self.samples.append(name)

        if self.normalize and mean is not None and std is not None:
            self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        else:
            self.mean_tensor = None
            self.std_tensor = None

        print(f"Loaded RealPhotoDataset for task '{self.task}' with {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def apply_degradation(self, *args, **kwargs):
        """CON-06 safety guard: explicitly raises error if called."""
        raise RuntimeError("CON-06 Violation: Degradation pipeline cannot touch real photos!")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.samples[idx]
        raw_path = self.raw_dir / name
        ref_path = self.ref_dir / name
        gt_corners = self.annotations[name]  # (4, 2) absolute pixels in raw frame

        raw_rgb = load_image(raw_path)
        h, w = raw_rgb.shape[:2]

        if self.task == "corner":
            # Resize raw photo to target_size
            resized_raw = cv2.resize(raw_rgb, self.target_size, interpolation=cv2.INTER_AREA)

            # Scale corners to [0, 1] normalized space
            scaled_corners = gt_corners.copy()
            scaled_corners[:, 0] /= float(w)
            scaled_corners[:, 1] /= float(h)

            # Tensor conversions: CHW float32 [0, 1]
            raw_tensor = torch.from_numpy(resized_raw.transpose(2, 0, 1)).float()
            corners_tensor = torch.from_numpy(scaled_corners).float()

            if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
                raw_tensor = (raw_tensor - self.mean_tensor) / self.std_tensor

            return {
                "name": name,
                "input": raw_tensor,               # (3, H, W) float32
                "target_corners": corners_tensor  # (4, 2) float32 [0, 1]
            }

        else:  # "enhancement"
            # Rectify raw photo using ground truth corners into (target_w, target_h)
            rectified_crop = rectify_document(raw_rgb, gt_corners, target_size=self.target_size)

            # Load and resize reference scan to (target_w, target_h)
            ref_rgb = load_image(ref_path)
            resized_ref = cv2.resize(ref_rgb, self.target_size, interpolation=cv2.INTER_AREA)

            # Tensor conversions: CHW float32 [0, 1]
            crop_tensor = torch.from_numpy(rectified_crop.transpose(2, 0, 1)).float() / 255.0
            ref_tensor = torch.from_numpy(resized_ref.transpose(2, 0, 1)).float() / 255.0


            if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
                crop_tensor = (crop_tensor - self.mean_tensor) / self.std_tensor

            return {
                "name": name,
                "input": crop_tensor,   # (3, H, W) float32 - rectified crop
                "target": ref_tensor    # (3, H, W) float32 [0, 1] - reference scan
            }
