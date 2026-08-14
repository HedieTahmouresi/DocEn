"""Corner Detection Networks Training Script (Approach A & Approach B).

Trains CornerRegNet (Approach A: Coordinate Regression) and CornerHeatmapNet (Approach B: Heatmap Regression)
under identical data streams, budgets, and hardware environments per [REQ-30], [REQ-31], and ADR-007.

Enforces:
- [CON-01] No pre-built architectures
- [CON-02] No pretrained weights
- [CON-04] Zero dropout (dropout=0.0) and weight_decay=0.0
- ADR-007 Fair comparison protocol (same encoder backbone, same data stream, equal LR search effort)
- ADR-008 Gaussian heatmaps (sigma=8 px) and local 11x11 soft-argmax sub-pixel extraction

Usage:
    python train_corners.py --env colab_t4
    python train_corners.py --env colab_t4 --mirror-dir /content/drive/MyDrive/DocEn_runs --mirror-every 5
    python train_corners.py --env local_cpu --epochs 2 --samples-per-epoch 200   # Smoke run
"""

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import SyntheticTrainDataset, FrozenEvalDataset, RealPhotoDataset
from src.data.heatmaps import extract_corners_from_heatmaps, render_gaussian_heatmaps
from src.metrics.corners import compute_corner_metrics
from src.models.corner_net import CornerRegNet, CornerHeatmapNet
from src.utils.config import load_config, save_resolved_config
from src.utils.seeding import seed_everything, worker_init_fn


DEFAULT_CONFIGS = [
    "configs/exp/exp-009_corner_approach_a.yaml",
    "configs/exp/exp-010_corner_approach_b.yaml",
]

CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_mce_px",
    "val_mce_pct",
    "val_succ_1pct",
    "val_succ_2pct",
    "real_mce_px",
    "real_succ_1pct",
    "lr",
    "epoch_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Corner Detection Trainer (Approach A & Approach B)")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="Experiment YAML config paths")
    parser.add_argument("--env", default="colab_t4", choices=["local_cpu", "mx330", "colab_t4"])
    parser.add_argument("--resume", action="store_true", help="Continue arms from last.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (e.g. for smoke runs)")
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="Override samples per epoch")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow fallback if GPU requested but missing")
    parser.add_argument("--mirror-dir", type=str, default=None, help="Sync run dirs to this directory (Drive)")
    parser.add_argument("--mirror-every", type=int, default=5, help="Sync every N epochs")
    return parser.parse_args()


def enable_gpu_fast_paths():
    """Enable cuDNN benchmarking and TF32 math if CUDA is available."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(cfg: Dict[str, Any], allow_cpu_fallback: bool = False) -> Tuple[torch.device, bool]:
    """Resolve compute device and AMP settings from config."""
    requested = cfg.get("device", "cpu")
    use_amp = bool(cfg.get("amp", False))

    if requested == "cuda":
        if not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA device requested in config profile, but torch.cuda.is_available() is False. "
                    "Pass --allow-cpu-fallback to override."
                )
            print("WARNING: CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu"), False
        return torch.device("cuda:0"), use_amp
    return torch.device("cpu"), False


def mirror_run_dir(src_dir: Path, dest_parent: Optional[str]) -> None:
    """Sync local run directory to remote mirror (e.g. Google Drive) for durability."""
    if dest_parent is None:
        return
    dest_path = Path(dest_parent) / src_dir.name
    dest_path.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        rel = Path(root).relative_to(src_dir)
        target_dir = dest_path / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, target_dir / f)


class CornerArm:
    """Encapsulates one corner model arm (RegNet or HeatmapNet), optimizer, and metrics log."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device, use_amp: bool):
        self.cfg = cfg
        self.id = cfg.get("run", {}).get("experiment_id", "exp-009")
        self.name = cfg.get("run", {}).get("name", "corner_net")
        self.label = f"{self.id}_{self.name}"
        self.arch = cfg.get("model", {}).get("arch", "corner_reg")

        self.run_dir = Path(cfg.get("runs_root", "runs")) / self.label
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(cfg.get("run", {}).get("seed", 1337))

        base_ch = cfg.get("model", {}).get("base_channels", 64)
        levels = cfg.get("model", {}).get("levels", 4)
        dropout = cfg.get("model", {}).get("dropout", 0.0)

        assert dropout == 0.0, f"[CON-04] Dropout must be 0.0, got {dropout}"

        if self.arch == "corner_reg":
            self.model = CornerRegNet(base_channels=base_ch, levels=levels, dropout=dropout).to(device)
            self.criterion = nn.L1Loss().to(device)
        elif self.arch in ("corner_heatmap", "heatmap"):
            upsample = cfg.get("model", {}).get("upsample", "transpose")
            self.model = CornerHeatmapNet(base_channels=base_ch, levels=levels, upsample=upsample, dropout=dropout).to(device)
            self.criterion = nn.MSELoss().to(device)
        else:
            raise ValueError(f"Unknown corner arch: {self.arch}")

        optim_cfg = cfg.get("optim", {})
        lr = optim_cfg.get("lr", 1.0e-3)
        weight_decay = optim_cfg.get("weight_decay", 0.0)
        assert weight_decay == 0.0, f"[CON-04] weight_decay must be 0.0, got {weight_decay}"

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.epochs = optim_cfg.get("epochs", 40)
        self.grad_clip = optim_cfg.get("grad_clip", 1.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=lr / 100.0
        )
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

        self.device = device
        self.best_val_mce = float("inf")
        self.start_epoch = 1
        self.metrics_log: List[Dict[str, Any]] = []

    def train_step(self, batch: Dict[str, torch.Tensor], device: Optional[torch.device] = None, use_amp: bool = False) -> float:
        if device is None:
            device = self.device
        self.model.train()
        inputs = batch["input"].to(device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if self.arch == "corner_reg":
                targets = batch["target_corners"].to(device)  # [N, 4, 2]
                targets_flat = targets.view(targets.size(0), -1)  # [N, 8]
                preds = self.model(inputs)  # [N, 8]
                loss = self.criterion(preds, targets_flat)
            else:
                targets = batch["target_heatmaps"].to(device)  # [N, 4, 512, 512]
                preds = self.model(inputs)  # [N, 4, 512, 512]
                loss = self.criterion(preds, targets)

        self.scaler.scale(loss).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: Optional[torch.device] = None, use_amp: bool = False) -> Tuple[float, Dict[str, float]]:
        if device is None:
            device = self.device
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []


        for batch in loader:
            inputs = batch["input"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if self.arch == "corner_reg":
                    targets = batch["target_corners"].to(device)
                    targets_flat = targets.view(targets.size(0), -1)
                    preds = self.model(inputs)
                    loss = self.criterion(preds, targets_flat)

                    # Extract coordinates directly from predictions [N, 8]
                    coords_pred = preds.cpu().numpy()  # [N, 8]
                    coords_gt = targets_flat.cpu().numpy()  # [N, 8]
                else:
                    preds = self.model(inputs)
                    if "target_heatmaps" in batch:
                        targets = batch["target_heatmaps"].to(device)
                        loss = self.criterion(preds, targets)
                    else:
                        loss = torch.tensor(0.0, device=device)

                    # Extract coordinates via Argmax + Local Soft-Argmax
                    coords_pred, _ = extract_corners_from_heatmaps(preds.cpu().numpy(), window_size=11, normalize=True)
                    target_corners = batch["target_corners"]
                    if isinstance(target_corners, torch.Tensor):
                        coords_gt = target_corners.cpu().numpy().reshape(-1, 8)
                    else:
                        coords_gt = np.array(target_corners).reshape(-1, 8)


            total_loss += float(loss.item()) * inputs.size(0)
            all_preds.append(coords_pred)
            all_targets.append(coords_gt)

        avg_loss = total_loss / len(loader.dataset)
        preds_cat = np.concatenate(all_preds, axis=0)
        targets_cat = np.concatenate(all_targets, axis=0)

        metrics = compute_corner_metrics(preds_cat, targets_cat, canvas_size=(512, 512), normalized_input=True)
        return avg_loss, metrics

    def save_checkpoint(self, epoch: int, val_loss: float, val_mce: float, is_best: bool = False):
        ckpt_data = {
            "epoch": epoch,
            "arch": self.arch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "val_mce": val_mce,
            "config": self.cfg,
        }
        torch.save(ckpt_data, self.ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt_data, self.ckpt_dir / "best.pt")

    def load_checkpoint(self, ckpt_path: Path):
        if not ckpt_path.exists():
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_mce = ckpt.get("val_mce", float("inf"))
        print(f"Resumed {self.label} from epoch {ckpt['epoch']} (best_val_mce={self.best_val_mce:.2f} px)")


def main():
    args = parse_args()

    # Load resolved config for first arm to read environment profile
    primary_cfg = load_config(env=args.env, exp_file=args.configs[0])
    enable_gpu_fast_paths()

    device, use_amp = resolve_device(primary_cfg, allow_cpu_fallback=args.allow_cpu_fallback)
    print(f"Running Phase 06 Corner Training on device: {device} (amp={use_amp})")

    # Load all arm configs
    arm_configs = [load_config(env=args.env, exp_file=cfg_path) for cfg_path in args.configs]


    # Instantiate arms (assign to separate GPUs if multi-GPU available)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        print(f"Multi-GPU detected: {num_gpus} GPUs available. Assigning arms to separate GPUs:")
        arms = []
        for idx, cfg in enumerate(arm_configs):
            gpu_idx = idx % num_gpus
            arm_device = torch.device(f"cuda:{gpu_idx}")
            exp_id = cfg.get("run", {}).get("experiment_id", f"exp-00{idx+9}")
            print(f"  -> Arm {exp_id} ({cfg.get('model', {}).get('arch')}) assigned to {arm_device}")
            arms.append(CornerArm(cfg, device=arm_device, use_amp=use_amp))
    else:
        arms = [CornerArm(cfg, device=device, use_amp=use_amp) for cfg in arm_configs]


    # Prepare datasets & data loaders
    scans_dir = primary_cfg["clean_scans_dir"]
    bg_dir = primary_cfg["backgrounds_dir"]
    splits_file = primary_cfg["splits_file"]
    frozen_val_dir = Path(primary_cfg["frozen_dir"]) / "val"
    raw_photos_dir = primary_cfg["raw_photos_dir"]
    ref_scans_dir = primary_cfg["reference_scans_dir"]
    ann_file = primary_cfg["annotations_file"]

    samples_per_epoch = args.samples_per_epoch or primary_cfg.get("corner_samples_per_epoch", 2000)
    batch_size = primary_cfg.get("batch_size", 8)
    num_workers = primary_cfg.get("num_workers", 2)
    seed = primary_cfg.get("seed", 42)

    train_dataset = SyntheticTrainDataset(
        scans_dir=scans_dir,
        bg_dir=bg_dir,
        splits_file=splits_file,
        split="train",
        task="corner",
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        normalize=primary_cfg.get("data", {}).get("standardize", True),
        mean=primary_cfg.get("normalization", {}).get("mean"),
        std=primary_cfg.get("normalization", {}).get("std"),
    )

    val_dataset = FrozenEvalDataset(
        frozen_dir=frozen_val_dir,
        task="corner",
        normalize=primary_cfg.get("data", {}).get("standardize", True),
        mean=primary_cfg.get("normalization", {}).get("mean"),
        std=primary_cfg.get("normalization", {}).get("std"),
    )

    real_dataset = None
    if Path(raw_photos_dir).exists() and Path(ann_file).exists():
        real_dataset = RealPhotoDataset(
            raw_dir=raw_photos_dir,
            ref_dir=ref_scans_dir,
            ann_file=ann_file,
            task="corner",
            normalize=primary_cfg.get("data", {}).get("standardize", True),
            mean=primary_cfg.get("normalization", {}).get("mean"),
            std=primary_cfg.get("normalization", {}).get("std"),
        )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,

        generator=g,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    real_loader = None
    if real_dataset is not None:
        real_loader = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    epochs = args.epochs or primary_cfg.get("optim", {}).get("epochs", 40)

    # Resume arms if requested
    if args.resume:
        for arm in arms:
            arm.load_checkpoint(arm.ckpt_dir / "last.pt")

    start_epoch = max(arm.start_epoch for arm in arms)

    # Save resolved configs
    for arm in arms:
        save_resolved_config(arm.cfg, arm.run_dir / "config.yaml")
        csv_path = arm.run_dir / "metrics.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_FIELDS)

    print(f"Starting paired corner training for {len(arms)} arms over {epochs} epochs...")

    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        t0 = time.time()

        train_losses = {arm.label: 0.0 for arm in arms}

        # Shared data stream training pass
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            for arm in arms:
                loss_val = arm.train_step(batch, device=arm.device, use_amp=use_amp)
                train_losses[arm.label] += loss_val * batch["input"].size(0)

        epoch_secs = time.time() - t0

        # Epoch evaluation pass for both arms
        for arm in arms:
            avg_train_loss = train_losses[arm.label] / len(train_dataset)
            val_loss, val_metrics = arm.evaluate(val_loader, device=arm.device, use_amp=use_amp)

            real_mce_px = 0.0
            real_succ_1pct = 0.0
            if real_loader is not None:
                _, real_metrics = arm.evaluate(real_loader, device=arm.device, use_amp=use_amp)
                real_mce_px = real_metrics["mean_corner_error_px"]
                real_succ_1pct = real_metrics["success_rate_1pct"]


            curr_lr = arm.scheduler.get_last_lr()[0]
            arm.scheduler.step()

            val_mce = val_metrics["mean_corner_error_px"]
            is_best = val_mce < arm.best_val_mce
            if is_best:
                arm.best_val_mce = val_mce

            arm.save_checkpoint(epoch, val_loss, val_mce, is_best=is_best)

            # Log metrics row
            log_row = [
                epoch,
                round(avg_train_loss, 6),
                round(val_loss, 6),
                round(val_metrics["mean_corner_error_px"], 2),
                round(val_metrics["mean_corner_error_pct"], 3),
                round(val_metrics["success_rate_1pct"], 2),
                round(val_metrics["success_rate_2pct"], 2),
                round(real_mce_px, 2),
                round(real_succ_1pct, 2),
                f"{curr_lr:.2e}",
                round(epoch_secs, 1),
            ]

            with open(arm.run_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            print(
                f"Epoch {epoch:02d} [{arm.label}] | Train Loss: {avg_train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | Val MCE: {val_metrics['mean_corner_error_px']:.2f} px ({val_metrics['mean_corner_error_pct']:.2f}%) | "
                f"Succ@1%: {val_metrics['success_rate_1pct']:.1f}% | Real MCE: {real_mce_px:.2f} px"
            )

        if args.mirror_dir and (epoch % args.mirror_every == 0 or epoch == epochs):
            print(f"Mirroring runs to {args.mirror_dir}...")
            for arm in arms:
                mirror_run_dir(arm.run_dir, args.mirror_dir)

    print("Phase 06 Corner Detection training complete!")


if __name__ == "__main__":
    main()
