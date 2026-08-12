# Phase 05 — Enhancement Evaluation

## Objective

Produce every number and figure the enhancement half of the report needs: the synthetic table with
its baseline, the real-photo triplets, the OCR readability comparison, and the inference pipeline.

This is where the synthetic test set is **touched for the first and only time** (`[CON-07]`).

## Prerequisites

Phase 04 gate passed, winner selected **on validation**. Phase 01 complete (real photos,
annotations, transcripts).

## Requirements in force

`[REQ-24]`–`[REQ-29]`, `[REQ-44]`, `[REQ-45]`, `[REQ-47]`, `[REQ-48]` · `[CON-07]` ·
ADR-011 · `03-spec/evaluation-spec.md` · `02-research/evaluation-and-ocr.md`

---

## Tasks

### A. Metrics
1. `src/metrics/image.py`: PSNR and SSIM per `evaluation-spec.md` §1.
   **PSNR per image then averaged — never over a pooled batch.**
2. Sanity: PSNR of an image against itself is ∞ (guard with epsilon); SSIM is exactly 1.0.

### B. The required table — `[REQ-25]`, `[REQ-26]`
3. **Compute the no-model baseline first**, on the test bucket, through `BaselineDataset`.
4. Then training, validation, and test rows for the winning model.
5. `[REC]` baseline per split too — free, and it makes the improvement legible.
6. **Write the interpretation** (spec §3.3): large train-vs-test gap ⇒ overfitting; small gap with
   poor numbers everywhere ⇒ underfitting. State which yours is. The table without the reading is
   half the deliverable.

### C. Loss comparison table — `[REQ-45]`
7. All four variants on the same frozen sets, plus the zoomed-text figure from Phase 04.
8. Explain the expected PSNR/SSIM disagreement.

### D. Inference pipeline — `[REQ-29]`
9. `src/pipeline/enhance.py`: preprocess → predict → **resize back to original dimensions, convert
   to 8-bit** → visualise. All four steps are named in spec §3.4.
10. Must run on an arbitrary unseen rectified image. **Test with an odd aspect ratio, a greyscale
    JPEG, and an image with EXIF rotation** — the TAs will run this on photos you have never seen
    (`[REQ-49]`).

### E. Real-photo evaluation — `[REQ-27]`
11. **Triplets:** rectify each photo with the **annotated** corners → run the model → present
    (rectified input, your output, reference scan).
12. **OCR, matched-resolution protocol** (ADR-011 §5, `evaluation-spec.md` §5b): all three images
    pass through 512×512, then upsample identically to a common evaluation resolution (~2000 px long
    side).
13. **Report the full-resolution rectified input as a separate, labelled row.** If it OCRs better
    than the model output, **say so** — that is a real limitation of the resolution choice, not a
    protocol to revise after seeing the result.
14. **CER (primary)** on the 5 transcribed documents; per-document and mean. Normalisation:
    collapse whitespace, strip ends — **do not lowercase, do not strip punctuation.** State it.
15. **Confidence (secondary)** over all photos; drop `conf == -1` entries.
16. Fix and record the Tesseract `--psm`, language pack and version. **Run all OCR on one machine.**

### F. Analysis — `[REQ-27]`, `[REQ-28]`, `[REQ-48]`
17. Where does the model match the app, fall short, and **do better**? The spec asks all three.
    Plausible wins: commercial apps often over-sharpen and clip highlights, losing faint pencil
    marks or low-contrast figures.
18. State the fairness caveat: the reference has its own style — "different from CamScanner is not
    the same as worse than CamScanner."
19. **Discuss the synthetic-vs-real relationship** (`[REQ-28]`) — "that gap is the central challenge
    of this project."
20. Start the limitations list (`[REQ-48]`), including the **resolution ceiling** with the x-height
    arithmetic from `02-research/evaluation-and-ocr.md`.

---

## Gate

- [ ] Metric sanity checks pass (self-PSNR, self-SSIM)
- [ ] **No-model baseline computed first** and recorded
- [ ] Full four-row table produced; **model clearly beats the baseline on test**
- [ ] Interpretation written (overfitting vs underfitting)
- [ ] Loss-comparison table + zoomed figure complete
- [ ] Enhancement pipeline runs on unseen images, including the three edge cases
- [ ] Triplet figures produced for **all** real photos
- [ ] OCR run under the matched-resolution protocol; settings recorded
- [ ] Full-resolution raw row reported separately
- [ ] CER computed for 5 documents, per-document and mean
- [ ] Confidence computed over all photos, with the miscalibration caveat stated
- [ ] Test split touched **exactly once** — confirm no earlier run read it
- [ ] Every number exists in a `metrics.json` and is regenerable

---

## Failure modes

**Batch-pooled PSNR.** Because of the log, the mean of per-image PSNRs is not the PSNR of the pooled
MSE. Only the first is comparable to anything published.

**`data_range` mismatch.** Passing 255 on `[0,1]` tensors shifts every PSNR by ~48 dB. If your
numbers look absurd, check this first.

**SSIM parameter mismatch.** `skimage`'s default uniform 7×7 window gives a different number from
the conventional Gaussian 11×11 σ=1.5. Report the parameters you used.

**The baseline beating the model.** Not a curiosity — it is a bug signal. Check normalisation
direction, input/target alignment, and colour space before writing anything down.

**Naive OCR comparison.** Measures downsampling, not enhancement, and can invert the true ranking.
The matched-resolution protocol exists for this.

**Tesseract version drift.** Colab and the workstation may differ, which silently confounds the
comparison. Run all OCR on one machine.

**Reading confidence as accuracy.** Documented effect: enhanced images can score lower confidence
while achieving lower CER. That is why CER leads.

**Touching test more than once.** `[CON-07]`. If you find yourself re-running test after a tweak,
stop — that is validation's job, and it inflates the headline.

**Rectifying real photos with predicted corners here.** `[REQ-27]` specifies **annotated** corners
for this phase; predicted corners belong to Phase 08's comparison. Mixing them destroys the
`[REQ-41]` contrast.

---

## Skills

- `05-skills/eval-integrity.md` — **mandatory before computing any reported number**
- `05-skills/scope-guard.md` — resist "improving" the model after seeing test numbers

---

## Deliverables

| Artifact | Location |
|---|---|
| Metric implementations | `src/metrics/image.py`, `src/metrics/ocr.py` |
| Evaluation entry point | `evaluate.py` |
| Enhancement pipeline | `src/pipeline/enhance.py` |
| PSNR/SSIM table + baseline | `outputs/reports/` |
| Loss-comparison table | `outputs/reports/` |
| Real-photo triplets | `outputs/figures/p05_triplets_*.png` |
| OCR results | `metrics.json` + `outputs/reports/` |
| Limitations notes (started) | `outputs/reports/limitations.md` |
