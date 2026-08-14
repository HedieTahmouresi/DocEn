# Evaluation Specification

Every number that appears in the report, and exactly how to compute it.

**Requirements:** `[REQ-24]`–`[REQ-28]`, `[REQ-31]`, `[REQ-39]`, `[REQ-41]`, `[REQ-47]` ·
**Decisions:** ADR-011 · **Pitfalls:** `02-research/evaluation-and-ocr.md` ·
**Discipline:** `05-skills/eval-integrity.md`

**Rule:** no number appears in the report that does not exist in a `metrics.json` in a run
directory, regenerable by re-running `evaluate.py` against a checkpoint.

---

## 1. Metric definitions — pin these

### PSNR
```
per image:  mse = mean((pred - target)^2)          over H,W,C, in [0,1] space
            psnr = 10 * log10(1.0 / max(mse, eps))
report:     mean of per-image PSNR
```
- `data_range = 1.0` (ADR-009: metrics live in `[0,1]`).
- **Per image, then averaged.** Never over a pooled batch — because of the log, the two differ, and
  only the first is comparable to anything published.
- Clamp MSE with a small epsilon so identical images do not give `inf`.

### SSIM
Our implementation (ADR-010), parameters pinned:

| Parameter | Value |
|---|---|
| Window | 11×11 Gaussian |
| σ | 1.5 |
| K1, K2 | 0.01, 0.03 |
| `data_range` | 1.0 |
| Channels | computed per channel, averaged |

Per image, then averaged across images. **Report these parameters in the report** — an SSIM without
its window settings is not reproducible, and `skimage`'s default (uniform 7×7) gives a different
number.

### MS-SSIM
5 scales, weights `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`. Requires ≥161 px — fine at 512.
Used as a loss (ADR-006); report it too if convenient.

### Corner localisation error
```
per image:  mean over 4 corners of ||pred_i - gt_i||_2      absolute px at 512
report:     mean across images, AND as % of image diagonal (512x512 -> 724.1 px)
```
Report both forms. Pixels alone are meaningless without a resolution.

### Success rate
Fraction of images where **all four** corners are within a threshold. Report at two, both stated:
- **strict:** 1% of diagonal ≈ 7.2 px @512
- **lenient:** 2% of diagonal ≈ 14.5 px @512

### Quadrilateral IoU — `[REC]`
IoU between predicted and ground-truth quads. SDL-Net's primary metric. Captures whether the
resulting *crop* is usable, which mean corner distance does not: two predictions with equal mean
error can yield one good crop and one useless one.

---

## 2. The required enhancement table — `[REQ-25]`, `[REQ-26]`

| Split | PSNR | SSIM |
|---|---|---|
| **No-model baseline** (degraded input vs target, **test** bucket) | | |
| Training | | |
| Validation | | |
| Test | | |

**Compute the baseline first** (`[REQ-26]`). Two reasons beyond compliance: it calibrates the scale,
and it catches bugs — if the baseline beats the model, something is inverted, misaligned, or in the
wrong colour space.

`[REC]` Also compute the baseline per split; it costs nothing and makes the improvement legible.

**Interpretation the spec asks for (§3.3):** large train-vs-test gap ⇒ overfitting; small gap with
poor numbers everywhere ⇒ underfitting. **State which yours is** — the table without the reading is
half the deliverable.

Training-split metrics use freshly generated samples (that bucket is not frozen); use a fixed count
(e.g. 500) so the number is stable enough to compare across runs.

---

## 3. Loss-function comparison — `[REQ-45]`

One row per variant, all on the **same frozen val/test**, all at 512:

| Variant | PSNR | SSIM | Real-photo CER | Visual verdict |
|---|---|---|---|---|
| L-A MSE | | | | |
| L-B L1 | | | | |
| L-C L1+MS-SSIM | | | | |
| L-D +Sobel | | | | |

Plus a figure: the **same input** through all four models, zoomed on text so the sharpness
difference is visible.

**Expect PSNR and SSIM to disagree.** PSNR is a monotone function of MSE, so the L2-trained model is
directly optimising it and will often win on PSNR while looking worse. **Explain this in the
report** — noticing it is exactly the insight `[REQ-45]` rewards.

**Select the winner on validation, never test** (`[CON-07]`).

---

## 4. Corner comparison — `[REQ-31]`

| Model | Split | Mean err (px @512) | Mean err (% diag) | Success @1% | Success @2% | IoU |
|---|---|---|---|---|---|---|
| Approach A | synthetic test | | | | | |
| Approach A | **real photos** | | | | | |
| Approach B | synthetic test | | | | | |
| Approach B | **real photos** | | | | | |

Synthetic and real **always reported as a pair**. The gap between them is the headline finding
(`[REQ-28]`).

`[REQ-31]` asks three questions; answer each with evidence:

| Question | Evidence |
|---|---|
| More accurate? | The table above |
| More robust to unusual viewpoints? | **Stratify the synthetic test set** by perspective severity and page scale; report error per stratum. Plus the hardest real photos. |
| Easier to train? | Epochs to converge, LR sensitivity, stability, restarts — **recorded during training**, not reconstructible after |

Plus **failure-case visualisations from both models** (`[REQ-31]`), not just the loser's.

---

## 5. Real-photo enhancement evaluation — `[REQ-27]`

No clean target exists, so PSNR/SSIM against ground truth is impossible. Two deliverables:

### 5a. Qualitative triplets
For each real photo: **(rectified input, your output, reference scan)** side by side. Rectify using
the **annotated** corners (`[REQ-27]`).

Answer in the write-up: where does the model match the app, where does it fall short, and **where
does it do better** — the spec asks all three. Plausible wins for a gentler restoration: commercial
apps often over-sharpen and clip highlights, losing faint pencil marks or low-contrast figures.

**Fairness caveat, from the spec itself:** the reference has its own style — aggressive contrast,
whitened background, sharpening. "Different from CamScanner is not the same as worse than
CamScanner."

### 5b. OCR readability — the matched-resolution protocol

**Rule: every image in the comparison passes through the same resolution pipeline.**

For each photo, build three images:
1. **Rectified input** → 512×512 → upsample to eval resolution
2. **Model output** (512×512) → upsample to eval resolution
3. **Reference scan** → 512×512 → upsample to eval resolution

Same interpolation for all three. Eval resolution: long side ~2000 px, fixed in config. This
isolates *enhancement* as the only difference — see `02-research/evaluation-and-ocr.md` for why the
naive version would measure downsampling instead and could invert the true ranking.

**Report a fourth row separately, clearly labelled: the full-resolution rectified input.** It
answers "would the user have been better off skipping our model?" If the answer is uncomfortable,
that is a limitation to disclose under `[REQ-48]` — not a protocol to revise after seeing the
result.

**Metrics:**
- **CER (primary):** hand-transcribe **5 documents**; `CER = Levenshtein / len(reference)`.
  Normalisation: collapse whitespace, strip leading/trailing. **Do not** lowercase, **do not** strip
  punctuation — those are real OCR errors. State the normalisation. Report per-document *and* mean;
  with 5 documents one bad case dominates the average.
- **Confidence (secondary):** mean word confidence via `pytesseract.image_to_data` over **all**
  photos; drop `conf == -1` entries (layout boxes, not words).

**State the caveat:** Tesseract's confidence is calibrated on its training image statistics, and
enhanced images shift those, so confidence can fall while CER improves. This is why CER leads.

**Tesseract settings — fix and record:** page segmentation mode (`--psm 6` suits a rectified page),
language pack, and version. **Run all OCR on one machine** — versions differ between Colab and the
workstation, and that would silently confound the comparison.

---

## 6. Dropout ablation — `[REQ-38]`, `[REQ-39]`

| Model | Variant | Synthetic val | Real photos | **Gap** |
|---|---|---|---|---|
| Enhancement | no dropout | | | |
| Enhancement | dropout | | | |
| Corner (winner) | no dropout | | | |
| Corner (winner) | dropout | | | |

The **Gap** column is the point. `[REQ-39]`: "*does the gap between synthetic validation scores and
real-photo test scores shrink?*" **Answer it explicitly in a sentence**, for both models. A table
without that sentence does not satisfy the requirement.

---

## 7. Bonus — `[REQ-41]`

| Rectification | OCR CER | OCR confidence | Qualitative |
|---|---|---|---|
| **Annotated** corners | | | |
| **Predicted** corners | | | |

The difference "tells you exactly how much corner errors cost the enhancement stage" — interpret it,
do not just tabulate it.

**Check corner ordering first.** Spec §7 hint: wrong order flips or rotates the page, and the
enhancement result then looks catastrophic for reasons unrelated to the enhancement network. Render
the colour-coded overlay (`00-project/conventions.md` §8) before believing any bonus number.

---

## 8. Output format

```
runs/<exp-id>_<slug>/
├── config.yaml           the exact resolved config
├── metrics.json          every number, machine-readable
├── history.csv           per-epoch training log
├── checkpoints/
└── figures/
```

`metrics.json` should carry the experiment id, git commit, `frozen_version`, and the split each
number came from. That is what makes "which run produced this table row?" answerable six weeks
later.
