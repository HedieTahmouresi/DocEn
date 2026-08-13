"""Train every arm of the loss ablation over one shared data stream.

`train.py` trains one model. Running it four times regenerates the same 80,000 synthetic
samples four times over, and on Colab's two vCPUs the generator — not the T4 — is what
sets the pace. This script builds the data once and steps all four models on every batch.

Two consequences, one practical and one methodological:

- **Wall clock.** The redundant generation disappears and the run becomes GPU-bound
  instead of loader-bound, so the GPU stops idling between batches.
- **Rigour.** ADR-006 asks for "identical in every other respect, one variable at a time".
  Separate runs achieve that only up to RNG luck in the data stream. Here every arm sees
  *the same batches in the same order*, from *identical initial weights* — a paired
  comparison, which is the strongest form of the claim the report needs to make.

Output is byte-compatible with `train.py`: one run directory per arm, same `metrics.csv`,
same checkpoint layout. `scripts/evaluate_ablation.py` consumes either without changes.

    python train_ablation.py --env colab_t4
    python train_ablation.py --env colab_t4 --resume          # continue all four arms
    python train_ablation.py --env local_cpu --epochs 2 --samples-per-epoch 200   # smoke
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from src.data.freeze import get_git_commit_hash
from src.losses.composite import EnhancementLoss
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config, save_resolved_config
from src.utils.seeding import seed_everything
from train import (
    build_model,
    enable_gpu_fast_paths,
    make_datasets,
    make_loaders,
    mirror_run_dir,
    resolve_device,
    save_checkpoint,
    to_device,
)

DEFAULT_CONFIGS = [
    "configs/exp/exp-005_enh_mse.yaml",
    "configs/exp/exp-006_enh_l1.yaml",
    "configs/exp/exp-007_enh_l1msssim.yaml",
    "configs/exp/exp-008_enh_l1msssim_sobel.yaml",
]

CSV_FIELDS = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr", "epoch_seconds"]


def parse_args():
    parser = argparse.ArgumentParser(description="Shared-data loss ablation trainer")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--env", default="colab_t4", choices=["local_cpu", "mx330", "colab_t4"])
    parser.add_argument("--resume", action="store_true", help="Continue every arm from its last.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs (smoke runs)")
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="Override (smoke runs)")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--mirror-dir", type=str, default=None, help="Sync run dirs here periodically")
    parser.add_argument("--mirror-every", type=int, default=5)
    return parser.parse_args()


def assert_comparable(configs: List[Dict[str, Any]], paths: List[str]) -> None:
    """The ablation is only meaningful if the loss is the only difference.

    This is the phase-04 gate item "identical seed, architecture, schedule, batch size and
    frozen sets" checked at launch instead of by eye. Sharing a data stream makes an
    accidental mismatch *more* dangerous, not less: four arms stepping on one loader must
    genuinely want the same loader.
    """
    reference, ref_path = configs[0], paths[0]

    for cfg, path in zip(configs[1:], paths[1:]):
        for key in ("model", "optim"):
            if cfg.get(key) != reference.get(key):
                raise ValueError(f"'{key}' differs between {ref_path} and {path}; arms are not comparable")
        for key in ("resolution", "samples_per_epoch", "frozen_version", "standardize"):
            if cfg.get("data", {}).get(key) != reference.get("data", {}).get(key):
                raise ValueError(f"data.{key} differs between {ref_path} and {path}")
        if cfg.get("run", {}).get("seed") != reference.get("run", {}).get("seed"):
            raise ValueError(f"run.seed differs between {ref_path} and {path}")

    loss_types = [cfg.get("loss", {}).get("type") for cfg in configs]
    if len(set(loss_types)) != len(loss_types):
        raise ValueError(f"Two arms share a loss type: {loss_types}. Nothing is being ablated.")

    # [CON-04] applies to every arm, not just whichever one gets inspected.
    for cfg, path in zip(configs, paths):
        if cfg.get("model", {}).get("dropout", 0.0) != 0.0:
            raise ValueError(f"[CON-04] dropout must be 0.0 in Phase 04 ({path})")
        if cfg.get("optim", {}).get("weight_decay", 0.0) != 0.0:
            raise ValueError(f"[CON-04] weight_decay must be 0.0 in Phase 04 ({path})")


class Arm:
    """One model, optimizer, scheduler, loss and run directory."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device, use_amp: bool, memory_format):
        self.cfg = cfg
        self.id = cfg.get("run", {}).get("experiment_id", "exp-000")
        self.name = cfg.get("run", {}).get("name", "enh_model")
        self.label = f"{self.id}_{self.name}"
        self.loss_type = cfg.get("loss", {}).get("type", "l1_msssim")

        self.run_dir = Path(cfg.get("runs_root", "runs")) / self.label
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Seed immediately before construction so every arm starts from *identical*
        # weights. Without this the arms differ by their initialisation as well as by
        # their loss, and the ablation has two variables.
        seed_everything(cfg.get("run", {}).get("seed", 1337))
        self.model = build_model(cfg).to(device=device, memory_format=memory_format)

        loss_cfg = cfg.get("loss", {})
        self.criterion = EnhancementLoss(
            loss_type=self.loss_type,
            alpha=loss_cfg.get("alpha", 0.84),
            sobel_weight=loss_cfg.get("sobel_weight", 0.1),
        ).to(device)

        optim_cfg = cfg.get("optim", {})
        lr = optim_cfg.get("lr", 1.0e-3)
        assert optim_cfg.get("weight_decay", 0.0) == 0.0, "[CON-04] weight_decay must be 0.0"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        assert self.optimizer.defaults["weight_decay"] == 0.0, "[CON-04] optimizer weight_decay non-zero"

        self.epochs = optim_cfg.get("epochs", 40)
        self.grad_clip = optim_cfg.get("grad_clip", 1.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=lr / 100.0
        )
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

        self.best_val_loss = float("inf")
        self.best_val_psnr = 0.0
        self.start_epoch = 1
        self.metrics_log: List[Dict[str, Any]] = []
        self.failed: Optional[str] = None

        # Per-epoch accumulators
        self._train_loss_sum = 0.0
        self._train_batches = 0

    # -- checkpointing ------------------------------------------------------------

    def resume(self, device: torch.device) -> None:
        last = self.ckpt_dir / "last.pt"
        if not last.exists():
            print(f"  {self.label}: no last.pt, starting from scratch")
            return
        # weights_only defaults to True from torch 2.6; our checkpoints carry a config dict.
        ckpt = torch.load(last, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_val_psnr = ckpt.get("best_val_psnr", 0.0)

        json_file = self.run_dir / "metrics.json"
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    self.metrics_log = [r for r in json.load(f) if int(r["epoch"]) < self.start_epoch]
            except (json.JSONDecodeError, KeyError, ValueError):
                print(f"  {self.label}: could not parse metrics.json, starting the JSON log fresh")

        print(f"  {self.label}: resumed at epoch {self.start_epoch} (best val loss {self.best_val_loss:.4f})")

    def checkpoint(self, epoch: int, val_loss: float, val_psnr: float, use_amp: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if use_amp else None,
            "best_val_loss": self.best_val_loss,
            "best_val_psnr": self.best_val_psnr,
            "config": self.cfg,
        }
        save_checkpoint(self.ckpt_dir / "last.pt", state)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_psnr = val_psnr
            state["best_val_loss"] = val_loss
            state["best_val_psnr"] = val_psnr
            save_checkpoint(self.ckpt_dir / "best.pt", state, weights_only=True)
            return True
        return False

    # -- logging ------------------------------------------------------------------

    def init_logs(self, resuming: bool) -> None:
        csv_file = self.run_dir / "metrics.csv"
        if not (resuming and csv_file.exists()):
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def log_epoch(self, row: Dict[str, Any]) -> None:
        self.metrics_log.append(row)
        with open(self.run_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics_log, f, indent=2)


def main():
    args = parse_args()

    configs = []
    for path in args.configs:
        cfg = load_config(env=args.env, exp_file=path)
        if args.epochs is not None:
            cfg.setdefault("optim", {})["epochs"] = args.epochs
        if args.samples_per_epoch is not None:
            cfg.setdefault("data", {})["samples_per_epoch"] = args.samples_per_epoch
        configs.append(cfg)

    assert_comparable(configs, args.configs)

    reference = configs[0]
    seed = reference.get("run", {}).get("seed", 1337)
    seed_everything(seed)

    device = resolve_device(reference, args.env, allow_cpu_fallback=args.allow_cpu_fallback)
    enable_gpu_fast_paths(device)

    amp_val = reference.get("amp") if "amp" in reference else reference.get("env", {}).get("amp", False)
    use_amp = bool(amp_val) and device.type == "cuda"
    memory_format = (
        torch.channels_last
        if reference.get("channels_last", False) and device.type == "cuda"
        else torch.contiguous_format
    )

    commit = get_git_commit_hash()
    for cfg in configs:
        cfg["amp"] = use_amp
        cfg["device"] = device.type
        cfg["git_commit"] = commit

    arms = [Arm(cfg, device, use_amp, memory_format) for cfg in configs]
    for arm in arms:
        save_resolved_config(arm.cfg, arm.run_dir)

    # Re-seed after model construction so the data stream does not depend on how many
    # arms were built.
    seed_everything(seed)
    train_dataset, val_dataset = make_datasets(reference, seed=seed, task="enhancement")
    train_loader, val_loader, batch_size = make_loaders(reference, train_dataset, val_dataset, device)

    if args.resume:
        print("Resuming arms:")
        for arm in arms:
            arm.resume(device)

    for arm in arms:
        arm.init_logs(resuming=args.resume)

    epochs = reference.get("optim", {}).get("epochs", 40)
    start_epoch = min(arm.start_epoch for arm in arms)
    if len({arm.start_epoch for arm in arms}) > 1:
        print(
            "WARNING: arms resumed at different epochs "
            f"({ {a.label: a.start_epoch for a in arms} }). They will be stepped together from "
            f"epoch {start_epoch}, so some will redo epochs. Prefer resuming an interrupted "
            "suite as a whole."
        )

    print(f"=== Loss ablation, {len(arms)} arms on one data stream ===")
    print(f"Device: {device} (AMP={use_amp}, channels_last={memory_format is torch.channels_last})")
    print(f"Seed: {seed}, Commit: {commit[:8]}, batch {batch_size}, epochs {start_epoch}..{epochs}")
    for arm in arms:
        print(f"  {arm.label:<32} loss={arm.loss_type}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_dataset.set_epoch(epoch)

        active = [a for a in arms if a.failed is None]
        if not active:
            raise RuntimeError("Every arm has failed; nothing left to train.")

        for arm in active:
            arm.model.train()
            arm._train_loss_sum = 0.0
            arm._train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch in pbar:
            inputs = to_device(batch["input"], device, memory_format)
            targets = to_device(batch["target"], device, memory_format)

            for arm in active:
                arm.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss = arm.criterion(arm.model(inputs), targets)

                if use_amp:
                    arm.scaler.scale(loss).backward()
                    arm.scaler.unscale_(arm.optimizer)
                    if arm.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(arm.model.parameters(), max_norm=arm.grad_clip)
                    arm.scaler.step(arm.optimizer)
                    arm.scaler.update()
                else:
                    loss.backward()
                    if arm.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(arm.model.parameters(), max_norm=arm.grad_clip)
                    arm.optimizer.step()

                arm._train_loss_sum += loss.item()
                arm._train_batches += 1

            pbar.set_postfix({a.loss_type: f"{a._train_loss_sum / a._train_batches:.4f}" for a in active})

        # One pass over the frozen validation set, forwarded through every arm.
        for arm in active:
            arm.model.eval()
        val_totals = {arm.label: {"loss": 0.0, "psnr": 0.0, "ssim": 0.0} for arm in active}
        num_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                inputs = to_device(batch["input"], device, memory_format)
                targets = to_device(batch["target"], device, memory_format)
                b = inputs.size(0)
                num_val += b

                for arm in active:
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = arm.model(inputs)
                        loss = arm.criterion(outputs, targets)
                    totals = val_totals[arm.label]
                    totals["loss"] += loss.item() * b
                    totals["psnr"] += calculate_psnr(outputs, targets) * b
                    totals["ssim"] += calculate_ssim(outputs, targets) * b

        epoch_sec = round(time.time() - t0, 2)
        print(f"Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.1f}s]")

        for arm in active:
            arm.scheduler.step()
            train_loss = arm._train_loss_sum / arm._train_batches
            totals = val_totals[arm.label]
            val_loss = totals["loss"] / num_val
            val_psnr = totals["psnr"] / num_val
            val_ssim = totals["ssim"] / num_val
            curr_lr = arm.optimizer.param_groups[0]["lr"]

            # A NaN kills this arm only. The other three keep the GPU busy and still
            # produce a usable comparison — losing all four to one bad loss would be
            # the expensive version of this failure (training-spec §3).
            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                arm.failed = f"non-finite loss at epoch {epoch} (train={train_loss}, val={val_loss})"
                print(f"  {arm.label:<32} FAILED: {arm.failed}")
                continue

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_psnr": round(val_psnr, 4),
                "val_ssim": round(val_ssim, 4),
                "lr": f"{curr_lr:.2e}",
                "epoch_seconds": epoch_sec,
            }
            arm.log_epoch(row)
            improved = arm.checkpoint(epoch, val_loss, val_psnr, use_amp)

            print(
                f"  {arm.label:<32} train {train_loss:.4f} | val {val_loss:.4f} | "
                f"PSNR {val_psnr:.2f} dB | SSIM {val_ssim:.4f}" + ("  <- best" if improved else "")
            )

        if args.mirror_dir and (epoch % args.mirror_every == 0 or epoch == epochs):
            for arm in arms:
                mirror_run_dir(arm.run_dir, Path(args.mirror_dir))
            print(f"  -> Mirrored {len(arms)} run directories to {args.mirror_dir}")

    print("\n=== Ablation complete ===")
    for arm in arms:
        if arm.failed:
            print(f"  {arm.label:<32} FAILED — {arm.failed}")
        else:
            print(
                f"  {arm.label:<32} best val loss {arm.best_val_loss:.4f} "
                f"(PSNR {arm.best_val_psnr:.2f} dB)"
            )
    print("\nNext: python -m scripts.evaluate_ablation")


if __name__ == "__main__":
    main()
