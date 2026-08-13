"""Training entry point [REQ-21].

Performs end-to-end training of enhancement and corner networks with:
- Config layering (base -> env -> exp)
- On-the-fly synthetic training generation
- Evaluation on frozen validation set (CON-07)
- Mixed-precision training (AMP float32 safe for MS-SSIM)
- Full checkpointing and resumption (--resume)
- Enforces [CON-04] (dropout == 0.0, weight_decay == 0.0 in Phase 04/06)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EnhancementNet, CornerRegNet, CornerHeatmapNet
from src.data.datasets import SyntheticTrainDataset, FrozenEvalDataset
from src.losses.composite import EnhancementLoss
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config, save_resolved_config
from src.utils.seeding import seed_everything, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Document Scanner Training Entry Point")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp/exp-001_enh_mse.yaml",
        help="Path or name of experiment YAML config",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="local_cpu",
        choices=["local_cpu", "mx330", "colab_t4"],
        help="Environment profile name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    return parser.parse_args()


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build model based on config specifications."""
    arch = cfg["model"]["arch"].lower()
    base_ch = cfg["model"].get("base_channels", 64)
    levels = cfg["model"].get("levels", 4)
    out_ch = cfg["model"].get("out_channels", 3)
    upsample = cfg["model"].get("upsample", "transpose")
    dropout = cfg["model"].get("dropout", 0.0)

    # Assert CON-04
    assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04, got {dropout}"

    if arch in ("unet", "enhancement"):
        return EnhancementNet(
            base_channels=base_ch,
            levels=levels,
            out_ch=out_ch,
            upsample=upsample,
            dropout=dropout,
        )
    elif arch in ("corner_reg", "approach_a"):
        return CornerRegNet(base_channels=base_ch, levels=levels, dropout=dropout)
    elif arch in ("corner_heatmap", "approach_b"):
        return CornerHeatmapNet(
            base_channels=base_ch, levels=levels, upsample=upsample, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = False,
    grad_clip: float = 1.0,
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / num_batches


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate model on frozen validation dataset."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        b_size = inputs.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * b_size
        total_psnr += calculate_psnr(outputs, targets) * b_size
        total_ssim += calculate_ssim(outputs, targets) * b_size
        num_samples += b_size

    return {
        "val_loss": total_loss / num_samples,
        "val_psnr": total_psnr / num_samples,
        "val_ssim": total_ssim / num_samples,
    }


def main():
    args = parse_args()

    # Load configuration
    cfg = load_config(env=args.env, exp_file=args.config)

    # Global seeding
    seed = cfg.get("run", {}).get("seed", 1337)
    seed_everything(seed)

    # Setup directories
    run_id = cfg.get("run", {}).get("experiment_id", "exp-001")
    exp_name = cfg.get("run", {}).get("name", "enh_model")
    run_dir = Path(cfg.get("runs_root", "runs")) / f"{run_id}_{exp_name}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    save_resolved_config(cfg, run_dir)

    # Set device
    device_str = cfg.get("env", {}).get("device", "cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            device_str = "cpu"
        else:
            try:
                _ = torch.zeros(1).cuda()
            except Exception:
                print("Warning: CUDA device detected but incompatible on this local hardware. Falling back to CPU.")
                device_str = "cpu"

    device = torch.device(device_str)

    use_amp = cfg.get("env", {}).get("amp", False) and device.type == "cuda"

    print(f"=== Starting Run {run_id}: {exp_name} ===")
    print(f"Device: {device} (AMP={use_amp}), Seed: {seed}")
    print(f"Run directory: {run_dir}")

    # Build model
    model = build_model(cfg).to(device)

    # Loss criterion
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "l1_msssim")
    alpha = loss_cfg.get("alpha", 0.84)
    sobel_w = loss_cfg.get("sobel_weight", 0.1)
    criterion = EnhancementLoss(loss_type=loss_type, alpha=alpha, sobel_weight=sobel_w).to(device)

    # Optimizer & Scheduler setup
    optim_cfg = cfg.get("optim", {})
    lr = optim_cfg.get("lr", 1.0e-3)
    weight_decay = optim_cfg.get("weight_decay", 0.0)

    # Assert CON-04
    assert weight_decay == 0.0, f"[CON-04] weight_decay must be 0.0 in Phase 04, got {weight_decay}"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # Assert weight_decay parameter inside optimizer defaults
    assert optimizer.defaults["weight_decay"] == 0.0, "[CON-04] Optimizer weight_decay is non-zero!"

    epochs = optim_cfg.get("epochs", 60)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100.0)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Load Datasets & DataLoaders
    data_cfg = cfg.get("data", {})
    data_root = Path(cfg.get("data_root", "data"))

    train_dataset = SyntheticTrainDataset(
        scans_dir=data_root / "clean_scans",
        bg_dir=data_root / "backgrounds",
        splits_file=data_root / "splits" / "splits.json",
        split="train",
        task="enhancement",
        samples_per_epoch=data_cfg.get("samples_per_epoch", 4000),
        target_size=(data_cfg.get("resolution", 512), data_cfg.get("resolution", 512)),
        generator_config=cfg.get("generator", None),
        seed=seed,
    )

    frozen_val_dir = Path(data_cfg.get("frozen_val_dir", data_root / "frozen" / "val"))
    val_dataset = FrozenEvalDataset(frozen_dir=frozen_val_dir, task="enhancement")

    num_workers = cfg.get("env", {}).get("num_workers", 2)
    batch_size = data_cfg.get("batch_size", 16)
    val_batch_size = max(32, batch_size * 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Checkpoint resume state
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_psnr = 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_psnr = checkpoint.get("best_val_psnr", 0.0)
        print(f"Resumed at epoch {start_epoch} with best val loss: {best_val_loss:.4f}")

    # CSV Logging setup
    csv_file = run_dir / "metrics.csv"
    csv_exists = csv_file.exists() and args.resume is not None
    csv_fieldnames = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr", "epoch_seconds"]

    if not csv_exists:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    metrics_log = []

    # Training loop
    grad_clip = optim_cfg.get("grad_clip", 1.0)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Update dataset epoch seed for variety
        train_dataset.epoch = epoch

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )

        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()
        epoch_sec = round(time.time() - t0, 2)
        curr_lr = optimizer.param_groups[0]["lr"]

        val_loss = val_metrics["val_loss"]
        val_psnr = val_metrics["val_psnr"]
        val_ssim = val_metrics["val_ssim"]

        print(
            f"Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | LR: {curr_lr:.2e}"
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_psnr, 4),
            "val_ssim": round(val_ssim, 4),
            "lr": f"{curr_lr:.2e}",
            "epoch_seconds": epoch_sec,
        }
        metrics_log.append(row)

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writerow(row)

        # JSON metrics update
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_log, f, indent=2)

        # Checkpointing
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_val_loss": best_val_loss,
            "best_val_psnr": best_val_psnr,
            "config": cfg,
        }

        # Save last checkpoint
        torch.save(ckpt_state, ckpt_dir / "last.pt")

        # Save best checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_psnr = val_psnr
            ckpt_state["best_val_loss"] = best_val_loss
            ckpt_state["best_val_psnr"] = best_val_psnr
            torch.save(ckpt_state, ckpt_dir / "best.pt")
            print(f"  -> Saved new best checkpoint (Val Loss: {best_val_loss:.4f}, PSNR: {best_val_psnr:.2f} dB)")

    print(f"=== Training Complete for {run_id}. Best Val Loss: {best_val_loss:.4f} ===")


if __name__ == "__main__":
    main()
