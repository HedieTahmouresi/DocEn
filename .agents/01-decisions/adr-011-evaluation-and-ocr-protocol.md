# ADR-011 — Metric Definitions and the Fair OCR Protocol

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Medium

## Context

The spec names the metrics (`[REQ-24]`, `[REQ-31]`) but leaves their exact definition open, and
several of the defaults are traps that change the reported numbers materially.

The OCR evaluation (`[REQ-27]`) has a deeper problem, uncovered while researching this project and
addressed by neither source document.

### The OCR resolution problem

`[REQ-27]` asks for OCR on three images: the rectified input, the model's output, and the commercial
reference scan. But the model works at 512×512 (ADR-002), and an A4 page rendered at 512 px gives
ordinary body text an x-height of **~2.6 px**, where Tesseract expects ~10 px.

That alone is a limitation. The trap is what happens if the three images are OCR'd at their
*natural* resolutions:

- the rectified **input** is available at full camera resolution — sharp text, bad lighting;
- the model **output** has passed through a 512×512 bottleneck — clean lighting, soft text;
- the **reference scan** comes from a commercial app at full resolution — sharp *and* clean.

Tesseract would then be measuring **downsampling**, not enhancement, and could plausibly score the
raw input *above* the enhanced output. That result would be an artifact of the protocol, not a
finding about the model.

There is a second, independently-documented pitfall: Tesseract's confidence scores are calibrated
against the image statistics it was trained on, and CNN-enhanced images can have altered intensity
distributions that yield *lower* confidence despite *lower* CER. Confidence and accuracy can move
in opposite directions.

## Decision

### 1. PSNR

- Computed on `[0,1]` tensors with `data_range=1.0` (ADR-009).
- **Per image, then averaged across images.** Never compute PSNR over a concatenated batch — MSE
  pooled across images is not the mean of per-image MSEs, and the resulting number is both wrong
  and quietly different from everyone else's.
- Averaged over RGB channels.
- Guard the degenerate case: identical images give infinite PSNR. Clamp MSE to a small epsilon.

### 2. SSIM

- Our implementation (ADR-010), parameters pinned there, `data_range=1.0`.
- Per image, averaged over channels, then averaged across images.
- **Report the parameters in the report.** SSIM computed with a uniform 7×7 window differs
  noticeably from a Gaussian 11×11 σ=1.5 window; an SSIM number without its settings is not
  reproducible.

### 3. The required table

`[REQ-25]`/`[REQ-26]` — four rows, baseline first:

| Row | Definition |
|---|---|
| **No-model baseline** | degraded input vs clean target, **test bucket** — compute first |
| Training | model output vs target, training split |
| Validation | model output vs target, frozen validation set |
| Test | model output vs target, frozen test set |

`[REC]`: also compute the baseline on train and validation. It costs nothing and makes the
improvement legible per split.

### 4. Corner metrics

- **Mean corner localisation error**: mean Euclidean distance between predicted and true corners,
  averaged over the 4 corners then over images. Reported **in px at 512×512** *and* as a **percentage
  of the image diagonal** (512×512 diagonal = 724.1 px), so it survives any resolution change and is
  comparable to published numbers.
- **Success rate**: fraction of images where **all four** corners fall within a threshold. Report at
  two thresholds and state both:
  - **strict:** 1% of diagonal ≈ 7.2 px @512
  - **lenient:** 2% of diagonal ≈ 14.5 px @512
- `[REC]` Also report **quadrilateral IoU** (predicted quad vs ground-truth quad). It is SDL-Net's
  primary metric, it is cheap, and it captures "is the crop right" better than corner distance —
  which is what actually matters downstream.
- Report on the synthetic test set **and** the real photos, always as a pair. The gap between them
  is the headline finding of this project (`[REQ-28]`, `[REQ-39]`).

**Do not compare absolute pixel numbers with the baseline notebook** in
`02-research/baseline-failure-analysis.md` — its resolution and threshold are undocumented. Its
*pattern* is the transferable evidence, not its magnitudes.

### 5. The fair OCR protocol

**Rule: every image in an OCR comparison passes through the same resolution pipeline.**

For each real photo, build all three images at a **common evaluation resolution**:

1. **Rectified input** — rectify with the annotated corners, resize to 512×512 (exactly as the model
   sees it), then upsample to the common evaluation resolution.
2. **Model output** — the 512×512 enhanced result, upsampled identically.
3. **Reference scan** — downsampled to 512×512, then upsampled identically.

Use the same interpolation everywhere. This isolates *enhancement* as the only difference, which is
what `[REQ-27]` is actually asking.

**Common evaluation resolution:** upsample so the long side is ~2000 px. Tesseract performs poorly
on small images regardless of content; giving all three the same generous canvas removes engine
scale-sensitivity as a confound. Fix the value in config and state it in the report.

**Additionally report, clearly labelled as a separate row: the full-resolution rectified input.**
This answers the honest question "would the user have been better off not using our model at all?"
If the raw full-resolution photo OCRs better than our 512-bottlenecked output, **say so** — that is
a genuine limitation of the chosen resolution and it belongs in `[REQ-48]`.

### 6. CER is primary; confidence is secondary

`[REQ-27]` permits either. Choose **both**, lead with CER:

- **CER (primary):** hand-transcribe the text of **5 documents** (spec says "a few"). Compute
  character error rate via Levenshtein distance / reference length, for all three images. Report
  per-document and mean.
- **Confidence (secondary):** mean word-level confidence from Tesseract over **all** photos, using
  only words above a confidence floor to exclude garbage detections.

**State the caveat in the report:** Tesseract's confidence is calibrated on its training image
statistics, and enhanced images can shift those statistics, so confidence can fall while CER
improves. That is why CER leads. This is a real, documented effect — and noting it is exactly the
kind of analysis `[REQ-27]` and `[REQ-48]` reward.

**Normalisation before CER:** collapse whitespace runs, strip leading/trailing space. Do **not**
lowercase and do **not** strip punctuation — case and punctuation errors are real OCR errors.
State the normalisation applied.

### 7. Reproducibility

Every reported number is written to a `metrics.json` in the run directory and is regenerable by
re-running `evaluate.py` against a checkpoint. **No number appears in the report that does not exist
in a `metrics.json`.** See `05-skills/eval-integrity.md`.

## Consequences

**Good.** Numbers are comparable, reproducible, and honestly framed. The OCR protocol measures the
intended thing. The resolution ceiling is surfaced as a finding rather than hidden as an artifact.

**Costs.** Hand-transcribing 5 documents is human time (~30 min). The protocol requires an extra
resampling pass per image — trivial.

**Risk.** OCR results may be weak in absolute terms at any whole-page resolution. Expected, and
handled by reporting *relative* improvement (input → output → reference) rather than absolute
accuracy, and by discussing the ceiling under `[REQ-48]`.

## Alternatives considered

- **OCR each image at its natural resolution.** The naive reading of `[REQ-27]`. Rejected: it
  measures downsampling, not enhancement, and could invert the true ranking.
- **Confidence only.** Cheaper — no transcription. Rejected: documented miscalibration on enhanced
  images makes it unreliable as the sole metric.
- **Word Error Rate instead of CER.** Reasonable, but harsher on documents with layout-driven word
  segmentation issues, and CER is the standard in the OCR-quality literature. `[REC]`: report WER
  too if it is free.
- **A second OCR engine (EasyOCR/PaddleOCR) as a cross-check.** Genuinely valuable for robustness,
  but the spec names Tesseract in its references. Optional stretch if Phase 05 finishes early.
