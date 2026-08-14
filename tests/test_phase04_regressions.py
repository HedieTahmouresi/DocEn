"""Regression tests for the defects found in the Phase 04 audit.

Each test here corresponds to a bug that produced no error message when it fired —
which is why it survived a passing 40-test suite and four GPU runs.
"""

import numpy as np
import pytest
import torch

from model import CornerHeatmapNet, CornerRegNet, EnhancementNet
from src.data.generator import SyntheticSampleGenerator
from src.data.normalization import denormalize, resolve, resolve_from_checkpoint
from src.utils.config import load_config


# --------------------------------------------------------------------------------------
# Output-head initialisation
# --------------------------------------------------------------------------------------

def _saturated_fraction(out: torch.Tensor, eps: float = 0.02) -> float:
    """Fraction of sigmoid outputs pinned against 0 or 1."""
    return float(((out < eps) | (out > 1.0 - eps)).float().mean())


@pytest.mark.parametrize(
    "build",
    [
        lambda: EnhancementNet(base_channels=16, levels=4),
        lambda: CornerHeatmapNet(base_channels=16, levels=4),
        lambda: CornerRegNet(base_channels=16, levels=4),
    ],
)
def test_sigmoid_heads_are_not_saturated_at_init(build):
    """The heads must start in the responsive part of the sigmoid.

    `kaiming_normal_(mode="fan_out", nonlinearity="relu")` on a head with 3, 4 or 8
    outputs draws weights around sqrt(2/out_features), which pushed the pre-sigmoid
    activations to 4-8 sigma. The sigmoid then starts hard against 0 or 1 where its
    derivative is ~0, and training spends its first epochs escaping the initialisation.
    """
    torch.manual_seed(0)
    model = build()
    model.train()  # BatchNorm uses batch statistics, as it does on step 1 of training

    out = model(torch.randn(4, 3, 64, 64))

    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    assert _saturated_fraction(out) < 0.25, (
        f"{type(model).__name__} starts {_saturated_fraction(out):.0%} saturated; "
        "the output head is being initialised as if it fed a ReLU."
    )
    assert out.std().item() > 1e-3, "Output has collapsed to a constant at init."


# --------------------------------------------------------------------------------------
# Generator config plumbing
# --------------------------------------------------------------------------------------

def test_generator_accepts_nested_or_flat_config(monkeypatch):
    """train.py passed the `generator:` block, freeze.py passed the whole config.

    The constructor only understood the second shape, so training silently ran on the
    hardcoded defaults while the frozen evaluation sets used base.yaml.
    """
    monkeypatch.setattr(SyntheticSampleGenerator, "_preload_assets", lambda self: None)

    gen_block = {
        "geometry": {"rotation_deg": [-5.0, 5.0]},
        "compression": {"jpeg_quality": [11, 12]},
    }

    nested = SyntheticSampleGenerator(
        ["scan.jpg"], ["bg.jpg"], config={"generator": gen_block, "heatmap_sigma": 4.0}
    )
    flat = SyntheticSampleGenerator(["scan.jpg"], ["bg.jpg"], config=gen_block)

    for gen in (nested, flat):
        assert list(gen.rotation_range_deg) == [-5.0, 5.0]
        assert list(gen.jpeg_quality_range) == [11, 12]

    # ADR-008 / [ASM-05] want sigma sweepable over {4, 8, 12}; it used to be hardcoded.
    assert nested.heatmap_sigma == 4.0
    assert flat.heatmap_sigma == 8.0  # falls back to the ADR-008 default


def test_generator_rejects_unknown_task(monkeypatch):
    monkeypatch.setattr(SyntheticSampleGenerator, "_preload_assets", lambda self: None)
    gen = SyntheticSampleGenerator(["scan.jpg"], ["bg.jpg"], config={})
    with pytest.raises(ValueError):
        gen.generate(task="both")


# --------------------------------------------------------------------------------------
# ADR-009 standardisation
# --------------------------------------------------------------------------------------

def test_denormalize_inverts_standardisation():
    mean = (0.8282, 0.8387, 0.8255)
    std = (0.1443, 0.1239, 0.1460)

    x = torch.rand(2, 3, 16, 16)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    standardised = (x - mean_t) / std_t
    assert torch.allclose(denormalize(standardised, mean, std), x, atol=1e-5)

    # And for a single un-batched CHW tensor
    assert torch.allclose(denormalize(standardised[0], mean, std), x[0], atol=1e-5)


def test_resolve_normalization_from_project_config():
    """base.yaml must carry statistics whenever standardisation is on (ADR-009)."""
    cfg = load_config()
    standardize, mean, std = resolve(cfg)
    if standardize:
        assert mean is not None and len(mean) == 3
        assert std is not None and len(std) == 3
        assert all(s > 0 for s in std)


def test_resolve_normalization_from_legacy_checkpoint():
    """A checkpoint predating standardisation saw [0, 1] input and must stay that way."""
    legacy = {"model": {"base_channels": 64}}  # no data.standardize key
    assert resolve_from_checkpoint(legacy) == (False, None, None)

    modern = {
        "data": {"standardize": True},
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    }
    standardize, mean, std = resolve_from_checkpoint(modern)
    assert standardize and mean == (0.5, 0.5, 0.5) and std == (0.25, 0.25, 0.25)


# --------------------------------------------------------------------------------------
# Environment profiles
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("env", ["local_cpu", "mx330", "colab_t4"])
def test_environment_profiles_exist_and_set_a_device(env):
    """A missing profile used to be skipped silently, turning GPU runs into CPU runs."""
    cfg = load_config(env=env)
    assert cfg["device"] in ("cpu", "cuda")
    assert isinstance(cfg["batch_size"], int) and cfg["batch_size"] > 0
    assert isinstance(cfg["num_workers"], int)
    if cfg["device"] == "cpu":
        assert cfg["amp"] is False, "AMP is meaningless on CPU"


def test_unknown_environment_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_config(env="does_not_exist")


# --------------------------------------------------------------------------------------
# Ablation comparability (phase-04 gate)
# --------------------------------------------------------------------------------------

EXP_CONFIGS = [
    "exp-005_enh_mse.yaml",
    "exp-006_enh_l1.yaml",
    "exp-007_enh_l1msssim.yaml",
    "exp-008_enh_l1msssim_sobel.yaml",
]


def test_ablation_runs_differ_only_in_the_loss():
    """ADR-006: identical seed, architecture, schedule, batch size and frozen sets.

    One variable at a time. A stray epoch count or resolution in one config would
    silently confound the graded comparison.
    """
    resolved = [load_config(env="colab_t4", exp_file=name) for name in EXP_CONFIGS]

    for key in ("model", "optim"):
        reference = resolved[0][key]
        for cfg in resolved[1:]:
            assert cfg[key] == reference, f"'{key}' differs across the ablation configs"

    for key in ("resolution", "samples_per_epoch", "frozen_version", "standardize"):
        values = {cfg["data"][key] for cfg in resolved}
        assert len(values) == 1, f"data.{key} differs across the ablation configs: {values}"

    assert len({cfg["run"]["seed"] for cfg in resolved}) == 1
    assert len({cfg["batch_size"] for cfg in resolved}) == 1
    assert len({cfg["loss"]["type"] for cfg in resolved}) == len(EXP_CONFIGS)

    # [CON-04] holds for every arm, not just the one that gets checked at runtime.
    for cfg in resolved:
        assert cfg["model"]["dropout"] == 0.0
        assert cfg["optim"]["weight_decay"] == 0.0


def test_shared_stream_trainer_accepts_the_real_configs():
    """`train_ablation.py` refuses to start unless the loss is the only difference."""
    from train_ablation import assert_comparable

    resolved = [load_config(env="colab_t4", exp_file=name) for name in EXP_CONFIGS]
    assert_comparable(resolved, EXP_CONFIGS)  # must not raise


def test_shared_stream_trainer_rejects_a_mismatched_arm():
    """A stray epoch count in one arm must stop the suite, not silently confound it."""
    from train_ablation import assert_comparable

    resolved = [load_config(env="colab_t4", exp_file=name) for name in EXP_CONFIGS]
    resolved[2]["optim"]["epochs"] += 1

    with pytest.raises(ValueError):
        assert_comparable(resolved, EXP_CONFIGS)


def test_shared_stream_trainer_rejects_duplicate_losses():
    """Two arms with the same loss means nothing is being ablated."""
    from train_ablation import assert_comparable

    resolved = [load_config(env="colab_t4", exp_file=name) for name in EXP_CONFIGS]
    resolved[1]["loss"]["type"] = resolved[0]["loss"]["type"]

    with pytest.raises(ValueError):
        assert_comparable(resolved, EXP_CONFIGS)


def test_ablation_schedule_is_long_enough_to_converge():
    """training-spec §4/§8: ~250 steps/epoch, converging somewhere around 40-60 epochs.

    The committed schedule was 20 x 1000 at batch 16 = 1,250 steps total, roughly an
    order of magnitude short of that.
    """
    cfg = load_config(env="colab_t4", exp_file=EXP_CONFIGS[0])
    steps_per_epoch = cfg["data"]["samples_per_epoch"] // cfg["batch_size"]
    total_steps = steps_per_epoch * cfg["optim"]["epochs"]

    assert steps_per_epoch >= 100, f"Only {steps_per_epoch} steps/epoch"
    assert total_steps >= 8000, f"Only {total_steps} optimiser steps for a 14.7M-param U-Net"
