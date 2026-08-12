"""
Dataset classes for evaluation and training.

Implements RealPhotoDataset per REQ-16 and enforces CON-06 (never run degradation pipeline on real photos).
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
        task: str = "enhancement"  # "enhancement" or "corner"
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.ref_dir = Path(ref_dir)
        self.ann_file = Path(ann_file)
        self.target_size = target_size
        self.task = task.lower()
        
        assert self.task in ("enhancement", "corner"), f"Invalid task: {task}"

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
            raw_tensor = torch.from_numpy(resized_raw.transpose(2, 0, 1)).float() / 255.0
            corners_tensor = torch.from_numpy(scaled_corners).float()

            return {
                "name": name,
                "input": raw_tensor,         # (3, H, W) float32 [0, 1]
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

            return {
                "name": name,
                "input": crop_tensor,   # (3, H, W) float32 [0, 1] - rectified crop
                "target": ref_tensor    # (3, H, W) float32 [0, 1] - reference scan
            }
