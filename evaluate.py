"""Phase 05 evaluation entry point [REQ-24].

Produces every number and figure the enhancement half of the report needs:

  1. The no-model baseline, computed FIRST, on the test bucket          [REQ-26]
  2. The four-row PSNR/SSIM table: baseline / train / val / test        [REQ-25], [REQ-47]
  3. The overfitting-vs-underfitting reading of that table              spec §3.3
  4. Real-photo triplets: rectified input | our output | reference      [REQ-27], [REQ-44]
  5. OCR readability under the matched-resolution protocol              [REQ-27], ADR-011 §5
  6. CER against the hand transcripts, per-document and mean            ADR-011 §6

Everything lands in `runs/<run>/metrics.json` so that every reported number traces to a
run directory (evaluation-spec §8, experiment-discipline rule 4).

    python evaluate.py --run runs/exp-008_enh_l1msssim_sobel
    python evaluate.py --run runs/exp-008_enh_l1msssim_sobel --skip-ocr   # no tesseract

[CON-07] The synthetic test split is read here and nowhere else. This is the final
evaluation phase, which is the one time it may be touched.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import EnhancementNet
from src.data.annotations import parse_coco_polygon_annotations
from src.data.datasets import FrozenEvalDataset, SyntheticTrainDataset
from src.data.freeze import get_git_commit_hash
from src.data.normalization import resolve_from_checkpoint
from src.geometry.homography import rectify_document
from src.metrics.baseline import evaluate_no_model_baseline
from src.metrics.image import calculate_psnr, calculate_ssim
from src.utils.config import load_config
from src.utils.io import load_image


# ADR-011 §5: every image in an OCR comparison passes through the same resolution
# pipeline, so that *enhancement* is the only difference between them. Long side ~2000 px
# because Tesseract degrades on small canvases regardless of content.
OCR_EVAL_LONG_SIDE = 2000
OCR_PSM = 6  # "a single uniform block of text" — right for a rectified page


def parse_args():
    p = argparse.ArgumentParser(description="Phase 05 enhancement evaluation [REQ-24]")
    p.add_argument("--run", required=True, help="Run directory, e.g. runs/exp-008_enh_l1msssim_sobel")
    p.add_argument("--env", default="local_cpu", choices=["local_cpu", "mx330", "colab_t4", "kaggle"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-samples", type=int, default=500,
                   help="Fresh samples for the training-split row (evaluation-spec §2)")
    p.add_argument("--skip-ocr", action="store_true", help="Skip OCR if tesseract is unavailable")
    p.add_argument("--max-triplets", type=int, default=30)
    return p.parse_args()


# ----------------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------------

def load_run(run_dir: Path, device: torch.device):
    """Load the trained model plus the input convention it was actually trained with.

    The architecture and the standardisation both come out of the checkpoint's own
    resolved config, never out of the current configs/ tree: a checkpoint trained before
    ADR-009 standardisation was wired up must still be evaluated on [0, 1] input, or its
    numbers are meaningless.
    """
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best.pt or last.pt under {run_dir / 'checkpoints'}")

    # weights_only defaults to True from torch 2.6; our checkpoints carry a config dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = EnhancementNet(
        base_channels=model_cfg.get("base_channels", 64),
        levels=model_cfg.get("levels", 4),
        out_ch=model_cfg.get("out_channels", 3),
        upsample=model_cfg.get("upsample", "transpose"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    standardize, mean, std = resolve_from_checkpoint(cfg)
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}, standardize={standardize})")
    return model, cfg, ckpt, (standardize, mean, std)


# ----------------------------------------------------------------------------------
# The required table — [REQ-25], [REQ-26]
# ----------------------------------------------------------------------------------

@torch.no_grad()
def score_loader(model, loader, device) -> Dict[str, float]:
    """Mean PSNR/SSIM over a loader, weighted per image (never batch-pooled)."""
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = model(inputs)
        b = inputs.size(0)
        psnr_sum += calculate_psnr(outputs, targets) * b
        ssim_sum += calculate_ssim(outputs, targets) * b
        n += b
    return {"psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1), "num_samples": n}


def build_split_rows(model, cfg, norm, args, device) -> Dict[str, Any]:
    """The four rows of [REQ-25]'s table, with the baseline computed first [REQ-26]."""
    standardize, mean, std = norm
    data_root = Path(cfg.get("data_root", "data"))
    resolution = cfg.get("data", {}).get("resolution", 512)
    target_size = (resolution, resolution)

    # --- Row 1: the no-model baseline, FIRST. [REQ-26] ---------------------------
    # "Compute it first. If your model's scores are not clearly above this line, it is
    # not earning its parameters." It is also a bug detector: a baseline that beats the
    # model means something is inverted, misaligned or in the wrong colour space.
    print("\n[1/4] No-model baseline (test bucket) — computed first per [REQ-26]")
    baseline_test = evaluate_no_model_baseline(split="test", batch_size=args.batch_size)
    print("      ...and on validation, for legibility of the improvement (ADR-011 §3 [REC])")
    baseline_val = evaluate_no_model_baseline(split="val", batch_size=args.batch_size)

    rows: Dict[str, Any] = {
        "baseline_test": {
            "psnr": baseline_test["baseline_psnr"],
            "ssim": baseline_test["baseline_ssim"],
            "num_samples": baseline_test.get("num_samples"),
        },
        "baseline_val": {
            "psnr": baseline_val["baseline_psnr"],
            "ssim": baseline_val["baseline_ssim"],
            "num_samples": baseline_val.get("num_samples"),
        },
    }

    # --- Row 2: training split, on freshly generated samples ---------------------
    # That bucket is not frozen (it is generated on the fly), so a fixed count keeps the
    # number stable enough to compare across runs — evaluation-spec §2.
    print(f"\n[2/4] Training split ({args.train_samples} fresh samples)")
    train_ds = SyntheticTrainDataset(
        scans_dir=data_root / "clean_scans",
        bg_dir=data_root / "backgrounds",
        splits_file=data_root / "splits" / "splits.json",
        split="train",
        task="enhancement",
        samples_per_epoch=args.train_samples,
        target_size=target_size,
        generator_config=cfg,
        seed=cfg.get("seed", 42),
        normalize=standardize,
        mean=mean,
        std=std,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows["train"] = score_loader(model, train_loader, device)

    # --- Rows 3 and 4: the frozen sets -------------------------------------------
    for split in ("val", "test"):
        idx = 3 if split == "val" else 4
        print(f"\n[{idx}/4] Frozen {split} set")
        ds = FrozenEvalDataset(
            frozen_dir=data_root / "frozen" / split,
            task="enhancement",
            target_size=target_size,
            normalize=standardize,
            mean=mean,
            std=std,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        rows[split] = score_loader(model, loader, device)

    return rows


def interpret_table(rows: Dict[str, Any]) -> str:
    """Spec §3.3 asks for the reading, not just the table. State which one yours is."""
    train_ssim = rows["train"]["ssim"]
    test_ssim = rows["test"]["ssim"]
    base_ssim = rows["baseline_test"]["ssim"]
    gap = train_ssim - test_ssim
    lift = test_ssim - base_ssim

    if lift <= 0:
        return (
            f"WARNING: the model (test SSIM {test_ssim:.4f}) does not beat the no-model "
            f"baseline ({base_ssim:.4f}). Per [REQ-26] it is not earning its parameters, and "
            "per eval-integrity §3 this is a bug signal — check normalisation direction, "
            "input/target alignment and colour space before reporting anything."
        )
    if gap > 0.05:
        return (
            f"Overfitting. Train SSIM {train_ssim:.4f} vs test {test_ssim:.4f} is a gap of "
            f"{gap:.4f}; the model has learned the training distribution more than the task. "
            f"It still beats the no-model baseline by {lift:+.4f} SSIM."
        )
    return (
        f"Neither strongly overfitting nor underfitting. Train SSIM {train_ssim:.4f} vs test "
        f"{test_ssim:.4f} is a gap of only {gap:.4f}, and the model clears the no-model "
        f"baseline by {lift:+.4f} SSIM. A small gap with good numbers on both sides is the "
        "outcome on-the-fly generation is designed to produce: the network effectively never "
        "sees the same image twice, so there is little fixed training set to memorise."
    )


# ----------------------------------------------------------------------------------
# Real photos — [REQ-27]
# ----------------------------------------------------------------------------------

def to_ocr_canvas(img_rgb: np.ndarray, resolution: int = 512) -> np.ndarray:
    """Put an image through the model's resolution pipeline, then onto the OCR canvas.

    ADR-011 §5. Every image in the comparison takes this identical path — down to
    512x512, then up to the common evaluation resolution with the same interpolation.
    OCR'd at their natural resolutions instead, the three images differ mostly in
    sharpness, and Tesseract would be measuring *downsampling* rather than enhancement;
    it could plausibly rank the raw input above the enhanced output, which would be an
    artefact of the protocol rather than a finding about the model.

    img_rgb: (H, W, 3) uint8 RGB
    returns: (L, L, 3) uint8 RGB where L = OCR_EVAL_LONG_SIDE
    """
    small = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (OCR_EVAL_LONG_SIDE, OCR_EVAL_LONG_SIDE), interpolation=cv2.INTER_CUBIC)


@torch.no_grad()
def enhance_rgb(model, img_rgb: np.ndarray, norm, device, resolution: int = 512) -> np.ndarray:
    """Run the enhancement network over an RGB uint8 image. Returns RGB uint8 at 512."""
    standardize, mean, std = norm
    resized = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div_(255.0)
    if standardize and mean is not None and std is not None:
        m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        x = (x - m) / s
    out = model(x.unsqueeze(0).to(device))[0].clamp(0.0, 1.0).cpu().numpy()
    return (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def evaluate_real_photos(model, cfg, norm, args, device) -> Dict[str, Any]:
    """Triplets [REQ-27.1] plus matched-resolution OCR readability [REQ-27.2]."""
    raw_dir = Path(cfg["raw_photos_dir"])
    ref_dir = Path(cfg["reference_scans_dir"])
    ann_file = Path(cfg["annotations_file"])
    transcripts_dir = Path(cfg.get("transcripts_dir", Path(cfg["real_photos_dir"]) / "transcripts"))
    resolution = cfg.get("data", {}).get("resolution", 512)

    if not (raw_dir.exists() and ann_file.exists()):
        print(f"Real photos or annotations missing ({raw_dir}); skipping [REQ-27].")
        return {}

    raw_files = sorted(f for f in raw_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    annotations = parse_coco_polygon_annotations(ann_file, active_filenames=[f.name for f in raw_files])

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_photo: List[Dict[str, Any]] = []
    triplets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    ocr_available = not args.skip_ocr
    if ocr_available:
        try:
            from src.metrics.ocr import compute_cer, run_ocr_on_image
        except ImportError as exc:
            print(f"OCR unavailable ({exc}); continuing with triplets only.")
            ocr_available = False

    for raw_path in raw_files:
        name = raw_path.name
        if name not in annotations:
            continue
        ref_path = ref_dir / name
        if not ref_path.exists():
            print(f"  {name}: no matching reference scan, skipped")
            continue

        raw_rgb = load_image(raw_path)
        corners = annotations[name]

        # [REQ-27] rectifies with the ANNOTATED corners. Predicted corners belong to
        # Phase 08's comparison; mixing them here would destroy the [REQ-41] contrast.
        rect_512 = rectify_document(raw_rgb, corners, target_size=(resolution, resolution))
        enhanced = enhance_rgb(model, rect_512, norm, device, resolution)
        reference = load_image(ref_path)

        if len(triplets) < args.max_triplets:
            triplets.append((name, rect_512, enhanced, cv2.resize(
                reference, (resolution, resolution), interpolation=cv2.INTER_AREA)))

        record: Dict[str, Any] = {"photo": name}

        if ocr_available:
            # All three take the identical resolution path — ADR-011 §5.
            texts = {
                "rectified_input": run_ocr_on_image(to_ocr_canvas(rect_512, resolution), psm=OCR_PSM),
                "model_output": run_ocr_on_image(to_ocr_canvas(enhanced, resolution), psm=OCR_PSM),
                "reference_scan": run_ocr_on_image(to_ocr_canvas(reference, resolution), psm=OCR_PSM),
            }
            # Reported separately and clearly labelled: the honest answer to "would the
            # user have been better off skipping our model?" If this wins, that is a real
            # limitation of the 512 resolution choice and belongs in [REQ-48] — it is not
            # a protocol to revise after seeing the result (eval-integrity §5).
            long_side = max(raw_rgb.shape[:2])
            full_rect = rectify_document(raw_rgb, corners, target_size=(long_side, long_side))
            texts["rectified_input_full_res"] = run_ocr_on_image(full_rect, psm=OCR_PSM)

            for key, (text, conf) in texts.items():
                record[f"{key}_conf"] = conf
                record[f"{key}_chars"] = len(text.strip())

            transcript_path = transcripts_dir / f"{raw_path.stem}.txt"
            if transcript_path.exists():
                gt = transcript_path.read_text(encoding="utf-8")
                record["has_transcript"] = True
                for key, (text, _conf) in texts.items():
                    record[f"{key}_cer"] = compute_cer(text, gt)
            else:
                record["has_transcript"] = False

        per_photo.append(record)
        print(f"  {name}: done")

    save_triplet_figure(triplets, fig_dir / "p05_triplets_real.png")

    summary = summarise_ocr(per_photo)
    return {"per_photo": per_photo, "summary": summary,
            "protocol": {
                "matched_resolution": True,
                "model_resolution": resolution,
                "ocr_eval_long_side": OCR_EVAL_LONG_SIDE,
                "tesseract_psm": OCR_PSM,
                "cer_normalisation": "collapse whitespace runs, strip ends; case and punctuation preserved",
            }}


def summarise_ocr(per_photo: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean CER over the transcribed documents; mean confidence over all photos."""
    keys = ["rectified_input", "model_output", "reference_scan", "rectified_input_full_res"]
    out: Dict[str, Any] = {}
    transcribed = [r for r in per_photo if r.get("has_transcript")]

    for key in keys:
        cers = [r[f"{key}_cer"] for r in transcribed if f"{key}_cer" in r]
        confs = [r[f"{key}_conf"] for r in per_photo if f"{key}_conf" in r]
        out[key] = {
            "mean_cer": float(np.mean(cers)) if cers else None,
            "cer_documents": len(cers),
            "mean_confidence": float(np.mean(confs)) if confs else None,
            "confidence_photos": len(confs),
        }

    inp = out.get("rectified_input", {}).get("mean_cer")
    mod = out.get("model_output", {}).get("mean_cer")
    ref = out.get("reference_scan", {}).get("mean_cer")
    if inp is not None and mod is not None:
        out["did_enhancement_beat_the_raw_photo"] = bool(mod < inp)
        out["cer_improvement_vs_input"] = float(inp - mod)
    if mod is not None and ref is not None:
        out["cer_gap_to_commercial_app"] = float(mod - ref)

    # ADR-011 §6: Tesseract's confidence is calibrated on its own training image
    # statistics, and CNN-enhanced images shift those. Confidence can fall while CER
    # improves, which is why CER leads and confidence is secondary.
    out["caveat"] = (
        "CER is primary. Tesseract confidence is calibrated against its training image "
        "statistics; enhanced images shift those, so confidence can fall while CER improves."
    )
    return out


def save_triplet_figure(triplets, out_path: Path) -> None:
    """(rectified input, our output, reference scan) per photo — [REQ-27.1], [REQ-44]."""
    if not triplets:
        print("No triplets to plot.")
        return
    n = len(triplets)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.7 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    titles = ["Rectified input", "Our output", "Reference scan (commercial)"]
    for row, (name, rect, enhanced, reference) in enumerate(triplets):
        for col, img in enumerate((rect, enhanced, reference)):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
        axes[row, 0].set_ylabel(name, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {n} triplets to {out_path}")


# ----------------------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    project_cfg = load_config(env=args.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {run_dir.name} on {device}")

    model, ckpt_cfg, ckpt, norm = load_run(run_dir, device)
    # Paths come from the project config; architecture and normalisation from the
    # checkpoint. Merged so the generator ranges used for the training-split row are the
    # ones this checkpoint was actually trained on.
    cfg = {**project_cfg, **{k: v for k, v in ckpt_cfg.items() if k in ("model", "data", "loss", "optim")}}
    for key in ("raw_photos_dir", "reference_scans_dir", "annotations_file", "real_photos_dir"):
        cfg.setdefault(key, project_cfg.get(key))

    rows = build_split_rows(model, cfg, norm, args, device)
    reading = interpret_table(rows)

    print("\n" + "=" * 78)
    print(f"[REQ-25] / [REQ-47] PSNR / SSIM — {run_dir.name}")
    print("=" * 78)
    print(f"{'Split':<34}{'PSNR (dB)':>12}{'SSIM':>10}{'n':>8}")
    print("-" * 78)
    order = [
        ("No-model baseline (test)", "baseline_test"),
        ("No-model baseline (val)", "baseline_val"),
        ("Training", "train"),
        ("Validation", "val"),
        ("Test", "test"),
    ]
    for label, key in order:
        r = rows[key]
        print(f"{label:<34}{r['psnr']:>12.4f}{r['ssim']:>10.4f}{str(r.get('num_samples','')):>8}")
    print("-" * 78)
    print(f"\nReading (spec §3.3): {reading}\n")

    real = evaluate_real_photos(model, cfg, norm, args, device)

    metrics = {
        "experiment_id": ckpt_cfg.get("run", {}).get("experiment_id", run_dir.name.split("_")[0]),
        "run_dir": str(run_dir),
        "checkpoint_epoch": ckpt.get("epoch"),
        "git_commit": get_git_commit_hash(),
        "frozen_version": ckpt_cfg.get("data", {}).get("frozen_version", "v1"),
        "resolution": ckpt_cfg.get("data", {}).get("resolution", 512),
        "loss_type": ckpt_cfg.get("loss", {}).get("type"),
        "ssim_settings": {
            "window": "11x11 Gaussian", "sigma": 1.5, "K1": 0.01, "K2": 0.03,
            "data_range": 1.0, "channels": "per channel, averaged",
        },
        "psnr_settings": {"data_range": 1.0, "per_image_then_averaged": True},
        "splits": rows,
        "interpretation": reading,
        "real_photos": real,
    }

    out_path = run_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "p05_enhancement_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {reports_dir / 'p05_enhancement_evaluation.json'}")


if __name__ == "__main__":
    main()
