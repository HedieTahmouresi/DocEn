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
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EnhancementNet, CornerRegNet, CornerHeatmapNet
from src.data.datasets import SyntheticTrainDataset, FrozenEvalDataset
from src.data.freeze import get_git_commit_hash
from src.data.normalization import resolve as resolve_normalization
from src.losses.composite import EnhancementLoss
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config, save_resolved_config
from src.utils.seeding import seed_everything, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Document Scanner Training Entry Point")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp/exp-005_enh_mse.yaml",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override optim.epochs (smoke runs only — never for a run that goes in the report)",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Override data.samples_per_epoch (smoke runs only)",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permit falling back to CPU when the profile asks for CUDA and it is unavailable",
    )
    parser.add_argument(
        "--mirror-dir",
        type=str,
        default=None,
        help="Copy the run directory here every --mirror-every epochs (e.g. a Drive folder). "
        "Lets checkpoints be written to fast local disk and synced periodically instead.",
    )
    parser.add_argument(
        "--mirror-every",
        type=int,
        default=5,
        help="Epoch interval for --mirror-dir (default 5)",
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


def to_device(
    tensor: torch.Tensor,
    device: torch.device,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    """Move a batch to the device in the layout the model expects."""
    tensor = tensor.to(device, non_blocking=True)
    if memory_format is torch.channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def enable_gpu_fast_paths(device: torch.device) -> None:
    """cuDNN autotuning. Worth ~10% and costs nothing here.

    `benchmark` picks the fastest conv algorithm per input shape by trying them once.
    It is only a win when the shapes are stable, which they are: `drop_last=True` makes
    every training batch identical in shape, and validation has at most two shapes.
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = False,
    grad_clip: float = 1.0,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        inputs = to_device(batch["input"], device, memory_format)
        targets = to_device(batch["target"], device, memory_format)

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
    memory_format: torch.memory_format = torch.contiguous_format,
) -> Dict[str, float]:
    """Evaluate model on frozen validation dataset."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        inputs = to_device(batch["input"], device, memory_format)
        targets = to_device(batch["target"], device, memory_format)
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


def save_checkpoint(path: Path, state: Dict[str, Any], weights_only: bool = False) -> None:
    """Write a checkpoint.

    `best.pt` is only ever read to *evaluate* a model, never to continue training, so it
    carries weights and config alone. That is ~59 MB instead of ~177 MB. When `runs/` is a
    Drive mount — which it must be, because Colab's local disk dies with the session — the
    difference is real wall-clock: at Drive's ~15 MB/s, per-epoch checkpointing was costing
    more than a third of an epoch's compute.
    """
    if weights_only:
        state = {k: v for k, v in state.items() if k in ("epoch", "model_state_dict", "config")}
    torch.save(state, path)


def mirror_run_dir(run_dir: Path, mirror_dir: Optional[Path]) -> None:
    """Copy a run directory's artefacts to a second location (typically Drive).

    Lets the run write checkpoints to fast local disk and pay the slow Drive write only
    every N epochs. A crash then costs at most N epochs of recompute instead of the run.
    """
    if mirror_dir is None:
        return
    target = Path(mirror_dir) / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for item in run_dir.rglob("*"):
        if item.is_file():
            dest = target / item.relative_to(run_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def resolve_device(cfg: Dict[str, Any], env_name: str, allow_cpu_fallback: bool = False) -> torch.device:
    """Resolve the device from the environment profile.

    A silent CPU fallback is how four "GPU" ablation runs previously came back unusable,
    so an unavailable CUDA device is an error unless the downgrade is asked for.
    """
    device_str = cfg.get("device") or cfg.get("env", {}).get("device", "cpu")
    if device_str != "cuda":
        return torch.device(device_str)

    reason = None
    if not torch.cuda.is_available():
        reason = "torch.cuda.is_available() is False"
    else:
        try:
            _ = torch.zeros(1).cuda()
        except Exception as exc:
            reason = f"CUDA is present but unusable on this hardware: {exc}"

    if reason is None:
        return torch.device("cuda")

    msg = f"Profile '{env_name}' requests device=cuda but {reason}."
    if not allow_cpu_fallback:
        raise RuntimeError(
            msg + " Re-run with --env local_cpu for a deliberate CPU run, "
            "or --allow-cpu-fallback to accept the downgrade."
        )
    print(f"WARNING: {msg} Falling back to CPU because --allow-cpu-fallback was given.")
    cfg["device"] = "cpu"
    return torch.device("cpu")


def make_datasets(cfg: Dict[str, Any], seed: int, task: str = "enhancement"):
    """Build the on-the-fly training set and the frozen validation set."""
    data_cfg = cfg.get("data", {})
    data_root = Path(cfg.get("data_root", "data"))
    resolution = data_cfg.get("resolution", 512)
    target_size = (resolution, resolution)

    # ADR-009: standardise the input, leave the target in [0, 1]. mean/std come from
    # the training split only (computed once in Phase 03 and stored in base.yaml).
    standardize, norm_mean, norm_std = resolve_normalization(cfg)

    train_dataset = SyntheticTrainDataset(
        scans_dir=data_root / "clean_scans",
        bg_dir=data_root / "backgrounds",
        splits_file=data_root / "splits" / "splits.json",
        split="train",
        task=task,
        samples_per_epoch=data_cfg.get("samples_per_epoch", 2000),
        target_size=target_size,
        generator_config=cfg,
        seed=seed,
        normalize=standardize,
        mean=norm_mean,
        std=norm_std,
    )

    frozen_val_dir = Path(data_cfg.get("frozen_val_dir", data_root / "frozen" / "val"))
    val_dataset = FrozenEvalDataset(
        frozen_dir=frozen_val_dir,
        task=task,
        target_size=target_size,
        normalize=standardize,
        mean=norm_mean,
        std=norm_std,
    )
    return train_dataset, val_dataset


def make_loaders(cfg: Dict[str, Any], train_dataset, val_dataset, device: torch.device):
    """Build the train and validation loaders."""
    data_cfg = cfg.get("data", {})
    num_workers = cfg.get("num_workers") if "num_workers" in cfg else cfg.get("env", {}).get("num_workers", 2)
    batch_size = cfg.get("batch_size") or data_cfg.get("batch_size", 8)
    val_batch_size = cfg.get("val_batch_size") or (batch_size * 2)

    # Persistent workers matter more than they look: without them the DataLoader forks
    # fresh workers every epoch, and each one re-runs `_preload_assets`, which decodes and
    # resizes 41 multi-megapixel scans plus 64 backgrounds. That was several seconds of
    # pure waste per epoch per run. The generator RNG stream now simply continues across
    # epochs, which is what "fresh samples every epoch" needs anyway — see
    # SyntheticTrainDataset.set_epoch. A deeper prefetch queue smooths the generator's
    # per-sample variance (a 91x91 shadow blur costs far more than a sample without one).
    worker_kwargs = (
        {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=(device.type == "cuda"),
        # A trailing batch of size 1 makes BatchNorm throw in train mode, and a short
        # final batch skews the epoch's BatchNorm statistics either way. It also keeps
        # every training batch the same shape, which is what makes cudnn.benchmark pay.
        drop_last=True,
        **worker_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        # The frozen set never changes, so keeping the workers alive keeps their
        # decoded-PNG cache warm instead of rebuilding it every epoch.
        **({"persistent_workers": True} if num_workers > 0 else {}),
    )
    return train_loader, val_loader, batch_size


def main():
    args = parse_args()

    # Load configuration
    cfg = load_config(env=args.env, exp_file=args.config)

    # CLI overrides, applied before anything reads the config so the resolved
    # config written to the run directory reflects what actually ran.
    if args.epochs is not None:
        cfg.setdefault("optim", {})["epochs"] = args.epochs
    if args.samples_per_epoch is not None:
        cfg.setdefault("data", {})["samples_per_epoch"] = args.samples_per_epoch

    # Global seeding
    seed = cfg.get("run", {}).get("seed", 1337)
    seed_everything(seed)

    # Setup directories
    run_id = cfg.get("run", {}).get("experiment_id", "exp-000")
    exp_name = cfg.get("run", {}).get("name", "enh_model")
    run_dir = Path(cfg.get("runs_root", "runs")) / f"{run_id}_{exp_name}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg, args.env, allow_cpu_fallback=args.allow_cpu_fallback)
    enable_gpu_fast_paths(device)

    amp_val = cfg.get("amp") if "amp" in cfg else cfg.get("env", {}).get("amp", False)
    use_amp = bool(amp_val) and device.type == "cuda"
    cfg["amp"] = use_amp

    # NHWC lets cuDNN reach the tensor-core conv kernels on Turing without a transpose
    # before every convolution. Mathematically identical, purely a memory layout.
    memory_format = (
        torch.channels_last
        if cfg.get("channels_last", False) and device.type == "cuda"
        else torch.contiguous_format
    )

    # training-spec §10: a metric without a commit cannot be reproduced.
    cfg["git_commit"] = get_git_commit_hash()

    # Written after device/AMP resolution so the run directory records what actually ran.
    save_resolved_config(cfg, run_dir)

    print(f"=== Starting Run {run_id}: {exp_name} ===")
    print(f"Device: {device} (AMP={use_amp}), Seed: {seed}, Commit: {cfg['git_commit'][:8]}")
    print(f"Run directory: {run_dir}")

    # Build model
    model = build_model(cfg).to(device=device, memory_format=memory_format)

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
    train_dataset, val_dataset = make_datasets(cfg, seed=seed, task="enhancement")
    train_loader, val_loader, _ = make_loaders(cfg, train_dataset, val_dataset, device)

    # Checkpoint resume state
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_psnr = 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming training from checkpoint: {resume_path}")
        # weights_only defaults to True from torch 2.6; our checkpoints carry the
        # resolved config dict, so loading them requires the full unpickler.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
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
    json_file = run_dir / "metrics.json"
    csv_exists = csv_file.exists() and args.resume is not None
    csv_fieldnames = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr", "epoch_seconds"]

    if not csv_exists:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    # On resume, carry the earlier epochs forward. Starting from [] used to truncate
    # metrics.json to the post-resume epochs only, which loses half a REQ-22 loss curve
    # every time a Colab session dies.
    metrics_log = []
    if args.resume and json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metrics_log = [row for row in json.load(f) if int(row["epoch"]) < start_epoch]
        except (json.JSONDecodeError, KeyError, ValueError):
            print(f"Warning: could not parse {json_file}; starting the JSON log fresh.")

    # Training loop
    grad_clip = optim_cfg.get("grad_clip", 1.0)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Re-seed the generator so this epoch composites fresh samples (REQ-11).
        train_dataset.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=grad_clip,
            memory_format=memory_format,
        )

        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            memory_format=memory_format,
        )

        scheduler.step()
        epoch_sec = round(time.time() - t0, 2)
        curr_lr = optimizer.param_groups[0]["lr"]

        val_loss = val_metrics["val_loss"]
        val_psnr = val_metrics["val_psnr"]
        val_ssim = val_metrics["val_ssim"]

        # MS-SSIM under AMP is the classic source of a mid-run NaN (training-spec §3).
        # Stopping here beats burning the rest of a Colab session on a dead run.
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch} (train={train_loss}, val={val_loss}). "
                "See .agents/05-skills/training-diagnostics.md before relaunching."
            )

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
        with open(json_file, "w", encoding="utf-8") as f:
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

        # last.pt carries the full state — it is what --resume reads
        save_checkpoint(ckpt_dir / "last.pt", ckpt_state)

        # best.pt is only ever read to evaluate, so it carries weights alone (~59 MB vs ~177 MB)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_psnr = val_psnr
            ckpt_state["best_val_loss"] = best_val_loss
            ckpt_state["best_val_psnr"] = best_val_psnr
            save_checkpoint(ckpt_dir / "best.pt", ckpt_state, weights_only=True)
            print(f"  -> Saved new best checkpoint (Val Loss: {best_val_loss:.4f}, PSNR: {best_val_psnr:.2f} dB)")

        if args.mirror_dir and (epoch % args.mirror_every == 0 or epoch == epochs):
            mirror_run_dir(run_dir, Path(args.mirror_dir))
            print(f"  -> Mirrored {run_dir.name} to {args.mirror_dir}")

    if args.mirror_dir:
        mirror_run_dir(run_dir, Path(args.mirror_dir))

    print(f"=== Training Complete for {run_id}. Best Val Loss: {best_val_loss:.4f} ===")


if __name__ == "__main__":
    main()
