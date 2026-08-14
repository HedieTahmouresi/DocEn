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
        """Update the epoch so each epoch draws fresh samples.

        Two paths, because the loaders use persistent workers:

        - `num_workers > 0`: the workers are created once, on the first iteration, and
          `worker_init_fn` seeds each of them from `self.epoch` at that moment. After
          that their RNG streams simply *continue* across epochs, which is exactly what
          fresh-samples-every-epoch means. Calling this method later has no effect on
          them, and does not need to.
        - `num_workers == 0`: `worker_init_fn` is never called at all, so without the
          re-seed below every epoch would replay the identical `samples_per_epoch`
          composites — an infinite-data pipeline silently collapsed into a fixed,
          memorisable set.
        """
        self.epoch = epoch
        self.rng = np.random.default_rng(self.seed + epoch)
        self.generator.rng = np.random.RandomState((self.seed + epoch) % (2 ** 32 - 1))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 HWC -> RGB float32 CHW in [0, 1] (conventions §3)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(rgb)).float().div_(255.0)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ADR-009 input standardisation, if configured."""
        if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
            return (x - self.mean_tensor) / self.std_tensor
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Only the outputs this task needs are rendered; the generator skips the rest.
        sample = self.generator.generate(task=self.task)

        w, h = self.target_size

        if self.task == "enhancement":
            return {
                "input": self._standardize(self._to_tensor(sample["enhance_input"])),
                "target": self._to_tensor(sample["enhance_target"]),
            }

        # "corner"
        corners_norm = sample["corners"].copy()
        corners_norm[:, 0] /= float(w)
        corners_norm[:, 1] /= float(h)

        return {
            "input": self._standardize(self._to_tensor(sample["composite"])),
            "target_corners": torch.from_numpy(corners_norm).float(),
            "target_heatmaps": torch.from_numpy(sample["heatmaps"]).float(),
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

    def _load_uint8(self, sample_id: str) -> Dict[str, np.ndarray]:
        """Decode (and memoise) only the PNGs this task needs, as uint8 RGB HWC.

        The cache deliberately holds uint8 rather than the assembled float tensors:
        500 samples of float32 CHW pairs is ~3 GB per worker, which does not fit
        alongside training on a Colab runtime. uint8 is 8x smaller, and the float
        conversion is negligible next to the PNG decode it saves.
        """
        cached = self._cache.get(sample_id)
        if cached is not None:
            return cached

        if self.task == "corner":
            imgs = {"composite": load_image(self.images_dir / f"{sample_id}_composite.png")}
        else:
            imgs = {
                "enh_input": load_image(self.images_dir / f"{sample_id}_enh_input.png"),
                "enh_target": load_image(self.images_dir / f"{sample_id}_enh_target.png"),
            }

        self._cache[sample_id] = imgs
        return imgs

    @staticmethod
    def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 HWC -> float32 CHW in [0, 1]."""
        return torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2, 0, 1))).float().div_(255.0)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ADR-009 input standardisation, if configured."""
        if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
            return (x - self.mean_tensor) / self.std_tensor
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        imgs = self._load_uint8(sample_id)

        # Corners are stored in absolute px at target_size; REQ-13 wants them in [0, 1].
        corners_abs = np.array(self.corners_data[sample_id], dtype=np.float32)  # (4, 2)
        w, h = self.target_size
        corners_norm = corners_abs.copy()
        corners_norm[:, 0] /= float(w)
        corners_norm[:, 1] /= float(h)

        if self.task == "corner":
            heatmaps = render_heatmaps(corners_abs, canvas_size=self.target_size, sigma=self.heatmap_sigma)
            return {
                "name": sample_id,
                "input": self._standardize(self._to_tensor(imgs["composite"])),
                "target_corners": torch.from_numpy(corners_norm).float(),
                "target_heatmaps": torch.from_numpy(heatmaps).float(),
            }

        return {
            "name": sample_id,
            "input": self._standardize(self._to_tensor(imgs["enh_input"])),
            "target": self._to_tensor(imgs["enh_target"]),
        }


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

    @staticmethod
    def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 HWC -> float32 CHW in [0, 1]."""
        return torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2, 0, 1))).float().div_(255.0)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ADR-009 input standardisation, if configured."""
        if self.normalize and self.mean_tensor is not None and self.std_tensor is not None:
            return (x - self.mean_tensor) / self.std_tensor
        return x

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

            # REQ-12: corners scale with the image, by the same factors.
            # REQ-13: normalised to [0, 1] by the same (w, h) convention the generator uses.
            scaled_corners = gt_corners.copy()
            scaled_corners[:, 0] /= float(w)
            scaled_corners[:, 1] /= float(h)

            return {
                "name": name,
                "input": self._standardize(self._to_tensor(resized_raw)),
                "target_corners": torch.from_numpy(scaled_corners).float(),  # (4, 2) [0, 1]
            }

        # "enhancement": rectify with the annotated corners (REQ-16), never degrade (CON-06)
        rectified_crop = rectify_document(raw_rgb, gt_corners, target_size=self.target_size)

        # Reference scan resized to the same size so the two are directly comparable (REQ-16).
        # It is a commercial baseline, NOT a training target (REQ-03).
        resized_ref = cv2.resize(load_image(ref_path), self.target_size, interpolation=cv2.INTER_AREA)

        return {
            "name": name,
            "input": self._standardize(self._to_tensor(rectified_crop)),
            "target": self._to_tensor(resized_ref),
        }
