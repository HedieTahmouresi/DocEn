"""Save full-resolution restored document samples to disk for inspection [REQ-44].

Writes, per sample: the degraded input, the clean target, and one restoration per loss
variant. The input and target are always written un-standardised — ADR-009's trap is
that a standardised tensor saved raw is a contrast-mangled image, not the degraded
photo the reader is being asked to compare against.
"""

from pathlib import Path

import numpy as np
import torch

from model import EnhancementNet
from src.data.datasets import BaselineDataset, FrozenEvalDataset
from src.data.normalization import resolve_from_checkpoint
from src.utils.io import save_image

RUNS = {
    "MSE": "exp-005_enh_mse",
    "L1": "exp-006_enh_l1",
    "L1_MSSSIM": "exp-007_enh_l1msssim",
    "L1_MSSSIM_Sobel": "exp-008_enh_l1msssim_sobel",
}


def load_variant(run_dir: Path, frozen_dir: Path, device: torch.device):
    """Load a checkpoint together with the eval dataset matching its training-time input."""
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        return None

    # weights_only defaults to True from torch 2.6; the checkpoint carries a config dict.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    model = EnhancementNet(
        base_channels=m_cfg.get("base_channels", 64),
        levels=m_cfg.get("levels", 4),
        out_ch=m_cfg.get("out_channels", 3),
        upsample=m_cfg.get("upsample", "transpose"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    standardize, mean, std = resolve_from_checkpoint(cfg)
    dataset = FrozenEvalDataset(
        frozen_dir=frozen_dir, task="enhancement", normalize=standardize, mean=mean, std=std
    )
    return model, dataset


def to_uint8(chw: np.ndarray) -> np.ndarray:
    """CHW float in [0, 1] -> HWC uint8 RGB, which is what save_image expects."""
    return (np.clip(chw.transpose(1, 2, 0), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def main(num_samples: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/figures/restored_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_dir = Path("data/frozen/val")
    display_dataset = BaselineDataset(frozen_dir=frozen_dir)  # always un-standardised

    variants = {}
    for name, run_name in RUNS.items():
        loaded = load_variant(Path("runs") / run_name, frozen_dir, device)
        if loaded is not None:
            variants[name] = loaded

    if not variants:
        print("No trained checkpoints found under runs/ — nothing to restore.")
        return

    with torch.no_grad():
        for idx in range(min(num_samples, len(display_dataset))):
            display = display_dataset[idx]
            save_image(to_uint8(display["target"].numpy()), out_dir / f"sample_{idx + 1}_target.png")
            save_image(
                to_uint8(display["input"].numpy()), out_dir / f"sample_{idx + 1}_degraded_input.png"
            )

            for name, (model, dataset) in variants.items():
                model_input = dataset[idx]["input"].unsqueeze(0).to(device)
                pred = model(model_input).squeeze(0).cpu().numpy()
                save_image(to_uint8(pred), out_dir / f"sample_{idx + 1}_restored_{name}.png")

    print(f"Saved individual restored document samples to {out_dir}/")


if __name__ == "__main__":
    main()
