# ADR-002 — Working Resolution: 512×512 for All Three Networks

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Medium (costs retraining)
**Decided by:** the human, explicitly, after seeing the alternatives.

## Context

Spec §2.2 step 3 says only: "Standardize all inputs and targets to your model's required input size
(e.g., 256×256 or 512×512 pixels)." The choice is ours.

Three sub-questions were on the table: the enhancement resolution, the corner-detector resolution,
and whether the loss ablation could be run more cheaply at a lower resolution than the final model.

**Enhancement.** The research report argues 512 is a floor: downsampling an A4 page to 256 px
compresses text strokes below one pixel, and once the Nyquist limit is breached the high-frequency
content is not recoverable by any decoder. That reasoning is sound.

**Corner detection.** The counter-argument for 256 is also sound: corner localisation keys on
global quadrilateral geometry, not on text detail. SDL-Net (arXiv:2310.00937) nonetheless uses
512×512 input with full-resolution 512×512 four-channel heatmaps, so 512 is well-precedented here.

**Ablation cost.** Running the four-way loss comparison at 256 would be ~4× cheaper, and the
*relative ordering* of loss functions transfers reliably across resolution.

## Decision

**512×512 everywhere.** Enhancement network, Approach A, Approach B, and every ablation including
the loss comparison and the dropout study. One resolution across the entire project.

Rationale for taking the more expensive option:
- **One number in the report.** No caveats, no "the ablation was run at a different resolution than
  the final model", no risk of a reviewer asking whether the loss ranking actually transferred.
- **Corner precision.** Predictions map back to full resolution more accurately; heatmap
  quantisation error is halved relative to 256. Matches SDL-Net's setup.
- **Direct evidence weighed.** The human has seen others struggle at 256 and chose to eliminate the
  risk rather than optimise the compute. That is a legitimate risk-versus-cost trade and it is
  recorded as the deciding factor.

**Consequence to hold onto:** a 512×512 render of a full A4 page still gives ordinary body text an
x-height of roughly **2.6 px**, where Tesseract wants ~10 px. **512 does not make OCR easy — it
makes it less impossible.** The OCR evaluation must therefore be structured to measure
*enhancement*, not resolution. That is ADR-011's matched-resolution protocol, and the resolution
ceiling itself is a required entry in the limitations discussion (`[REQ-48]`).

## Consequences

**Good.** Uniform, defensible, no cross-resolution comparisons anywhere. Best achievable text
fidelity within the spec's suggested sizes. Corner coordinates map back cleanly.

**Costs.**
- ~4× the GPU hours of a 256 plan across the whole matrix. Realistically several Colab sessions;
  checkpoint/resume (ADR-001) is not optional.
- 512² × 4-channel heatmap targets are 4 MB per sample in fp32. **Never render a full-frame
  Gaussian** — compute it in a ±3σ window and paste (ADR-008). Naive rendering is ~100× more work
  and will starve the GPU on Colab's 2 vCPUs.
- The MX330's 4 GB caps batch size at roughly 2–4 at this resolution. It is a smoke-test machine,
  not a training machine, which ADR-001 already assumed.
- Generator CPU cost rises. ADR-003's mitigations become load-bearing rather than nice-to-have.

## If compute becomes the binding constraint

**This is not a licence to change resolution unilaterally.** If the matrix genuinely will not fit,
the pre-analysed fallback — which still requires human approval via the deviation protocol — is,
in preference order:

1. Reduce the *number* of epochs or the samples-per-epoch, keeping 512. Cheapest quality loss.
2. Run only the **loss ablation** at 256, keeping every reported model at 512, and state the
   ablation resolution explicitly in the report.
3. Drop the corner detectors to 256 (the best-justified reduction on the merits).

Bring evidence: measured seconds/epoch, measured GPU utilisation, and the projected total.

## Alternatives considered

- **Enhance 512 / corners 256** — the original recommendation. Cheaper and technically defensible.
  Rejected in favour of uniformity and margin for error.
- **Everything at 256** — full matrix affordable, but ~1.3 px x-height makes text mush; the
  qualitative triplets against CamScanner and the OCR numbers are both graded, and both would
  suffer.
- **768 or 1024** — better text, but ~2.25–4× the cost of 512, beyond the sizes the spec names, and
  tighter CPU pressure. Out of scope; mention as future work in `[REQ-48]`.

## Notes for implementation

- MS-SSIM at 5 scales requires ≥161 px; 512 satisfies it comfortably (a 512 image reaches 32 px at
  the coarsest scale).
- With four downsampling stages, 512 → 32×32 at the bottleneck: enough spatial extent for the
  bottleneck to carry global illumination context, which is what shadow removal needs.
- Aspect ratio: source scans are not square. How pages are fitted into a square canvas is specified
  in `03-spec/synthetic-generator-spec.md` — it must be identical for training and inference, or
  the model sees a distribution shift at test time.
