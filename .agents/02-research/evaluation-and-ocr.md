# Evaluation Pitfalls and the OCR Problem

Background for ADR-011. Every item here changes a reported number, and most fail silently.

---

## Part 1 — Metric pitfalls

### PSNR must be averaged per image

`PSNR = 10·log₁₀(data_range² / MSE)`. Because of the logarithm, the mean of per-image PSNRs is
**not** the PSNR of the pooled MSE. Both are computable; only the first is what everyone reports.

Compute MSE per image → PSNR per image → average. A batch-pooled PSNR is dominated by the worst
images and is not comparable to any published number.

Also: `data_range` must match your tensor range. On `[0,1]` tensors it is 1.0; passing 255 by habit
shifts every PSNR by ~48 dB.

### SSIM parameters are not standard by default

`skimage.metrics.structural_similarity` defaults to a **uniform 7×7 window**. The SSIM of the
original paper — and of essentially every reported SSIM — uses an **11×11 Gaussian window with
σ=1.5**. These give different numbers on the same images.

To reproduce the conventional value, pass `gaussian_weights=True, sigma=1.5,
use_sample_covariance=False`. ADR-010 pins these, and the validation test compares like with like —
without those flags, a *correct* implementation fails the test, which is an easy afternoon to lose.

**Report the parameters alongside the number.** An SSIM without its window settings is not
reproducible.

### Multi-channel handling

SSIM on colour images is usually computed per channel and averaged. Some implementations convert to
greyscale first, which gives a different number. Pick one, state it, use it everywhere.

### The no-model baseline is not optional

`[REQ-26]` requires it and says to compute it **first**: PSNR/SSIM of the degraded input against
the clean target, on the test bucket. Spec: "If your model's scores are not clearly above this line,
it is not earning its parameters."

Two reasons this matters beyond compliance. It calibrates the scale — on synthetic document data,
degraded-vs-clean PSNR is often already respectable, so a model scoring 24 dB may be doing very
little. And it catches bugs: if the baseline scores *higher* than the model, something is wrong
(inverted normalisation, misaligned pair, wrong colour space) and you have found it before writing
the report.

### Corner error needs a resolution and a threshold

"Mean error 4.2 px" is meaningless alone. Report at the working resolution *and* as a fraction of
the image diagonal (512×512 → 724.1 px). Success rate must always carry its threshold in the same
sentence.

**Do not compare absolute pixel figures with the baseline notebook** (1.85 px / 107.44 px) — its
resolution and threshold are undocumented. Only the *pattern* transfers.

### Quadrilateral IoU is worth adding

`[REC]`, and SDL-Net's primary metric. Mean corner distance treats all errors equally; IoU captures
whether the resulting **crop** is right, which is what actually matters downstream. Two predictions
with the same mean corner error can produce a usable crop and a useless one. Cheap to compute
(shapely, or a polygon rasterisation).

---

## Part 2 — The OCR resolution problem

This is the significant issue neither source document addresses, and it shapes ADR-011's protocol.

### The arithmetic

Tesseract wants roughly **10 px of x-height** for reliable recognition; ~300 DPI is the usual
guidance. Consider an A4 page (297 mm tall) with 11 pt body text (x-height ≈ 1.5 mm):

| Whole-page render height | x-height |
|---|---|
| 256 px | **~1.3 px** |
| 512 px | **~2.6 px** |
| 1024 px | ~5.2 px |
| 2048 px | ~10.4 px |

**At 512 — this project's resolution (ADR-002) — body text has an x-height of about 2.6 px.** That
is roughly a quarter of what Tesseract needs. 512 makes OCR *less impossible* than 256; it does not
make it easy.

This is a genuine, structural limitation of the whole-page approach, and it belongs in the
limitations discussion `[REQ-48]` asks for.

### Why the naive comparison would be actively misleading

`[REQ-27]` asks for OCR on three images. At their natural resolutions they are not comparable:

| Image | Resolution available | Text | Lighting |
|---|---|---|---|
| Rectified input | full camera res (e.g. 3000 px) | **sharp** | bad |
| Model output | 512 px bottleneck | **soft** | good |
| Reference scan | full res, app-processed | sharp | good |

OCR'd as-is, Tesseract is largely measuring **resolution**, not enhancement — and could plausibly
rank the raw input above the enhanced output. That would be an artifact of the protocol, and
reporting it as a finding about the model would be wrong.

### The fix

**Put every image through the same resolution pipeline** (ADR-011 §5): each of the three passes
through 512×512, then all three are upsampled identically to a common evaluation resolution
(~2000 px long side). Enhancement becomes the only difference between them.

**And report the full-resolution raw input separately**, clearly labelled. It answers the honest
question — "would the user have been better off skipping our model?" — and if the answer is
uncomfortable, that is a limitation to disclose, not a protocol to quietly revise after seeing the
result (`05-skills/eval-integrity.md`).

### Why Tesseract confidence is not enough on its own

Documented effect: Tesseract's confidence scores are calibrated against the image statistics of its
training data. CNN-enhanced images have altered intensity distributions, and can therefore receive
**lower confidence despite lower CER** — better text extraction, worse-looking confidence.

Confidence and accuracy can move in opposite directions. `[REQ-27]` permits either; ADR-011 takes
both and leads with **CER**, with confidence as a secondary signal and the caveat stated in the
report.

### CER, concretely

`CER = Levenshtein(reference, hypothesis) / len(reference)`. 0 is perfect; values above 1 are
possible when the engine hallucinates extra text.

- Hand-transcribe **5 documents** (spec: "a few").
- Normalise before comparing: collapse whitespace runs, strip leading/trailing space.
- **Do not** lowercase and **do not** strip punctuation — case and punctuation errors are real OCR
  errors, and removing them flatters every system equally but hides the differences you want.
- State the normalisation you applied.
- Report per-document as well as the mean. With 5 documents, one bad case dominates the mean, and
  the per-document table is more informative than the average.

### Practical Tesseract notes

- Page segmentation mode matters. `--psm 6` ("assume a single uniform block of text") is usually
  right for a rectified page; the default `--psm 3` can mis-segment. Fix one mode for all images —
  varying it between conditions invalidates the comparison.
- Tesseract prefers dark text on light background. Our outputs already satisfy this.
- `pytesseract.image_to_data` gives per-word confidences; filter out `conf == -1` entries, which are
  layout boxes rather than words.
- Fix the language pack and version and record them. Different Tesseract versions give different
  results, and Colab's version may differ from the workstation's — **run all OCR on one machine.**

---

## Part 3 — What the report needs

Beyond raw numbers, `[REQ-27]`/`[REQ-28]`/`[REQ-48]` ask for interpretation:

- **The fairness caveat**, which the spec itself supplies: the commercial reference has its own
  style — aggressive contrast, whitened background, sharpening. "Different from CamScanner is not
  the same as worse than CamScanner." Judge on readability and on the triplet images.
- **The synthetic-vs-real relationship** (`[REQ-28]`): "a model can top the synthetic test set and
  still fail on real photos — that gap is the central challenge of this project." Report both, side
  by side, and interpret the gap.
- **The resolution ceiling**, from Part 2, as a named limitation with the arithmetic behind it.
- **Where the model beats the commercial app, if anywhere.** The spec explicitly asks. Plausible
  candidates: commercial apps often over-sharpen and clip highlights, losing faint pencil marks or
  low-contrast figures. A gentler restoration can preserve those.

---

## Sources

- Tesseract issue #3184 on supported image sizes and resolution guidance.
- Documented finding on Tesseract confidence miscalibration for CNN-enhanced images (see
  `source-index.md`).
- Wang et al. (2004), SSIM; Wang et al. (2003), MS-SSIM — window and parameter conventions.
- arXiv:2310.00937 (SDL-Net) — quadrilateral IoU as the primary localisation metric.
