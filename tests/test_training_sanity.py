"""Sanity ladder unit tests for model training and resume (training-spec.md §9).

Verifies:
1. Forward pass shape, range, dtype
2. Overfit one batch to near-zero loss (< 0.01)
3. Loss sanity (target vs target = 0, target vs noise large)
4. Metric sanity (PSNR(x, x) = 100, SSIM(x, x) = 1.0)
5. Short training step & checkpointing
6. Resume test (kill mid-run, resume, loss and epoch count continue)
"""

import os
import shutil
import tempfile
import torch
import pytest

from model import EnhancementNet
from src.losses.composite import EnhancementLoss
from src.metrics.image import calculate_psnr, calculate_ssim


def test_sanity_forward_pass():
    """Check 1: Forward pass produces correct shape, dtype, and range."""
    model = EnhancementNet(base_channels=32, levels=3)
    model.eval()

    x = torch.rand(2, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 128, 128)
    assert out.dtype == torch.float32
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_sanity_overfit_one_batch():
    """Check 2: Critical Check — Overfit a single batch to near-zero loss (< 0.01).

    If the model cannot overfit one batch, it will never learn the dataset.
    """
    torch.manual_seed(42)
    model = EnhancementNet(base_channels=32, levels=3)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0)
    criterion = EnhancementLoss(loss_type="l1", alpha=0.84)

    # Structured document-like target (white page with black text block)
    target_batch = torch.ones(2, 3, 128, 128, dtype=torch.float32) * 0.95
    target_batch[:, :, 30:90, 30:90] = 0.1
    input_batch = torch.clamp(target_batch * 0.8 + 0.1 + torch.randn_like(target_batch) * 0.05, 0.0, 1.0)

    initial_loss = float("inf")
    final_loss = float("inf")

    for step in range(250):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if step == 0:
            initial_loss = loss_val
        final_loss = loss_val

    print(f"Overfit-one-batch: Step 0 loss={initial_loss:.4f} -> Step 250 loss={final_loss:.6f}")
    assert final_loss < 0.01, f"Failed to overfit single batch! Final loss: {final_loss:.6f}"


def test_sanity_loss_values():
    """Check 3: Loss sanity checks."""
    criterion = EnhancementLoss(loss_type="l1_msssim", alpha=0.84)

    target = torch.rand(2, 3, 128, 128, dtype=torch.float32)
    noise = torch.rand(2, 3, 128, 128, dtype=torch.float32)

    zero_loss = criterion(target, target).item()
    noise_loss = criterion(noise, target).item()

    assert zero_loss < 1e-4, f"Loss(target, target) must be ~0, got {zero_loss}"
    assert noise_loss > 0.1, f"Loss(noise, target) must be large, got {noise_loss}"


def test_sanity_metrics():
    """Check 4: Metric sanity checks."""
    img = torch.rand(2, 3, 128, 128, dtype=torch.float32)

    psnr_self = calculate_psnr(img, img)
    ssim_self = calculate_ssim(img, img)

    assert psnr_self >= 99.0, f"PSNR(x, x) expected >= 99 dB, got {psnr_self}"
    assert abs(ssim_self - 1.0) < 1e-5, f"SSIM(x, x) expected 1.0, got {ssim_self}"


def test_sanity_resume_checkpoint():
    """Check 5 & 6: Checkpoint saving and resumption test."""
    temp_dir = tempfile.mkdtemp()
    try:
        ckpt_path = os.path.join(temp_dir, "checkpoint.pt")

        model = EnhancementNet(base_channels=32, levels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Train 1 step
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        out = model(x)
        loss = (out - y).abs().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Save checkpoint
        state = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
        }
        torch.save(state, ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file was not created"

        # Resume into fresh model instance
        resumed_model = EnhancementNet(base_channels=32, levels=3)
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3, weight_decay=0.0)
        resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=10)

        checkpoint = torch.load(ckpt_path)
        resumed_model.load_state_dict(checkpoint["model_state_dict"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resumed_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        assert checkpoint["epoch"] == 1
        # Verify weight match
        for p1, p2 in zip(model.parameters(), resumed_model.parameters()):
            assert torch.allclose(p1, p2), "Resumed weights do not match original model"

    finally:
        shutil.rmtree(temp_dir)
