"""Unit tests for model architectures and composite losses.

Verifies shapes, parameter bounds, zero-dropout constraints,
and composite loss variants.
"""

import pytest
import torch

from model import EnhancementNet, CornerRegNet, CornerHeatmapNet
from src.losses.composite import EnhancementLoss
from src.losses.sobel import SobelLoss


def test_enhancement_net_forward():
    """Verify EnhancementNet output shape, range, and dtype."""
    model = EnhancementNet(base_channels=64, levels=4)
    model.eval()

    x = torch.randn(2, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 512, 512), f"Expected (2, 3, 512, 512), got {out.shape}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output must be in [0, 1] due to sigmoid"


def test_corner_reg_net_forward():
    """Verify CornerRegNet output shape [N, 8] and range [0, 1]."""
    model = CornerRegNet(base_channels=64, levels=4)
    model.eval()

    x = torch.randn(2, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 8), f"Expected (2, 8), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output coords must be in [0, 1]"


def test_corner_heatmap_net_forward():
    """Verify CornerHeatmapNet output shape [N, 4, 512, 512] and range [0, 1]."""
    model = CornerHeatmapNet(base_channels=64, levels=4)
    model.eval()

    x = torch.randn(2, 3, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 4, 512, 512), f"Expected (2, 4, 512, 512), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Heatmaps must be in [0, 1]"


def test_dropout_zero_assertion():
    """Verify that dropout != 0.0 raises AssertionError under [CON-04]."""
    with pytest.raises(AssertionError):
        EnhancementNet(dropout=0.2)

    with pytest.raises(AssertionError):
        CornerRegNet(dropout=0.1)

    with pytest.raises(AssertionError):
        CornerHeatmapNet(dropout=0.5)


def test_parameter_counts():
    """Verify approximate parameter budget (~14.7M for EnhancementNet)."""
    enh_model = EnhancementNet(base_channels=64, levels=4)
    params = sum(p.numel() for p in enh_model.parameters())
    assert 10_000_000 < params < 18_000_000, f"Unexpected parameter count: {params}"


def test_composite_loss_variants():
    """Test all four loss variants (L-A, L-B, L-C, L-D)."""
    torch.manual_seed(42)
    pred = torch.rand(2, 3, 128, 128, dtype=torch.float32, requires_grad=True)
    target = torch.rand(2, 3, 128, 128, dtype=torch.float32)

    for loss_type in ["mse", "l1", "l1_msssim", "l1_msssim_sobel"]:
        criterion = EnhancementLoss(loss_type=loss_type, alpha=0.84, sobel_weight=0.1)
        loss = criterion(pred, target)

        assert isinstance(loss, torch.Tensor), f"Loss {loss_type} should return Tensor"
        assert loss.ndim == 0, f"Loss {loss_type} should be scalar"
        assert loss.item() > 0.0, f"Loss {loss_type} should be positive, got {loss.item()}"

        # Test backward pass gradient flow
        loss.backward()
        assert pred.grad is not None, f"Gradient must flow back to pred for {loss_type}"
        pred.grad.zero_()
