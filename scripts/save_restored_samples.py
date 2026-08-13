"""Save sample restored document images to disk for inspection."""

import os
import torch
import numpy as np
from pathlib import Path
from model import EnhancementNet
from src.data.datasets import FrozenEvalDataset
from src.utils.io import save_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/figures/restored_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dataset = FrozenEvalDataset(frozen_dir="data/frozen/val", task="enhancement")

    runs = {
        "MSE": Path("runs/exp-001_enh_mse"),
        "L1": Path("runs/exp-002_enh_l1"),
        "L1_MSSSIM": Path("runs/exp-003_enh_l1msssim"),
        "L1_MSSSIM_Sobel": Path("runs/exp-004_enh_l1msssim_sobel"),
    }

    models = {}
    for name, run_dir in runs.items():
        ckpt_path = run_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "checkpoints" / "last.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            m_cfg = ckpt.get("config", {}).get("model", {})
            base_ch = m_cfg.get("base_channels", 64)
            levels = m_cfg.get("levels", 4)
            out_ch = m_cfg.get("out_channels", 3)
            upsample = m_cfg.get("upsample", "transpose")
            m = EnhancementNet(base_channels=base_ch, levels=levels, out_ch=out_ch, upsample=upsample).to(device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            models[name] = m

    for idx in range(3):  # Save 3 sample documents
        batch = val_dataset[idx]
        inp = batch["input"].unsqueeze(0).to(device)
        tgt = batch["target"].numpy().transpose(1, 2, 0)  # [H, W, 3] in [0, 1]

        save_image(np.clip(tgt, 0.0, 1.0), str(out_dir / f"sample_{idx+1}_target.png"))
        save_image(np.clip(batch["input"].numpy().transpose(1, 2, 0), 0.0, 1.0), str(out_dir / f"sample_{idx+1}_degraded_input.png"))

        with torch.no_grad():
            for name, m in models.items():
                pred = m(inp).squeeze(0).cpu().numpy().transpose(1, 2, 0)
                pred_img = np.clip(pred, 0.0, 1.0)
                save_image(pred_img, str(out_dir / f"sample_{idx+1}_restored_{name}.png"))

    print(f"Saved individual restored document samples to {out_dir}/")


if __name__ == "__main__":
    main()
