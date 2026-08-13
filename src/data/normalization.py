"""Input standardisation helpers (ADR-009).

ADR-009 is deliberately asymmetric: the **input** is standardised with per-channel
statistics computed from the training split only, while the **target**, the model
output and every metric stay in `[0, 1]` with `data_range=1.0`.

Everything that builds a Dataset for a trained model must resolve the statistics the
same way, or inference silently runs on a different input distribution than training.
`resolve()` is the single place that decision lives.
"""

from typing import Any, Dict, Optional, Tuple

import torch

NormStats = Tuple[bool, Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]


def resolve(cfg: Dict[str, Any]) -> NormStats:
    """Return `(standardize, mean, std)` for a resolved project config.

    Raises if standardisation is requested but `configs/base.yaml` carries no
    statistics — silently training on un-standardised input is exactly the kind of
    invisible inconsistency ADR-009 exists to prevent.
    """
    norm_cfg = cfg.get("normalization") or {}
    data_cfg = cfg.get("data") or {}

    standardize = bool(data_cfg.get("standardize", True))
    mean = tuple(norm_cfg.get("mean", ())) or None
    std = tuple(norm_cfg.get("std", ())) or None

    if standardize and (mean is None or std is None):
        raise ValueError(
            "data.standardize is on but configs/base.yaml has no normalization.mean/std. "
            "Run `python -m src.data.compute_normalization`, or set data.standardize: false."
        )
    return standardize, mean, std


def resolve_from_checkpoint(ckpt_config: Dict[str, Any]) -> NormStats:
    """Resolve the statistics a checkpoint was *trained* with.

    Evaluation must use the training-time convention, not whatever the current
    `configs/base.yaml` happens to say. Checkpoints written before standardisation
    existed carry no `data.standardize` key; those models saw `[0, 1]` input, so the
    default here is off — the opposite of `resolve()`.
    """
    norm_cfg = ckpt_config.get("normalization") or {}
    data_cfg = ckpt_config.get("data") or {}

    standardize = bool(data_cfg.get("standardize", False))
    mean = tuple(norm_cfg.get("mean", ())) or None
    std = tuple(norm_cfg.get("std", ())) or None

    if standardize and (mean is None or std is None):
        return False, None, None
    return standardize, mean, std


def denormalize(
    x: torch.Tensor,
    mean: Optional[Tuple[float, ...]],
    std: Optional[Tuple[float, ...]],
) -> torch.Tensor:
    """Undo standardisation so a model *input* can be displayed or scored.

    ADR-009's trap: a standardised tensor rendered raw is a contrast-mangled image.
    Anything that shows the degraded input next to the output, or measures the
    no-model baseline against the target, must come back through here first.
    """
    if mean is None or std is None:
        return x
    shape = (-1, 1, 1) if x.dim() == 3 else (1, -1, 1, 1)
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(shape)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(shape)
    return x * std_t + mean_t
