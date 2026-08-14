# ADR-008 — Heatmap Representation and Coordinate Extraction

**Status:** PROVISIONAL — σ and the loss choice must be validated in Phase 06
**Date:** 2026-08-12 · **Reversibility:** High

## Context

`[REQ-30]` specifies Approach B only in outline: four heatmaps, one per corner, "each containing a
Gaussian blob centered on the true corner location", extracted at inference "with an argmax (or a
*soft-argmax* if you want the extraction to be differentiable)", trained "with a pixel-wise loss on
the heatmaps."

Everything else is ours: output resolution, σ, the exact loss, and the extraction rule.

Reference point: SDL-Net (arXiv:2310.00937) does document corner localisation with 512×512 input,
**full-resolution 512×512 four-channel heatmaps**, MSE loss, and peak extraction. That is close to
this project's setup — though it uses a pretrained MobileNetV2 encoder, which `[CON-02]` forbids
here.

## Decision

### Output resolution: full 512×512, four channels

Matches ADR-002 and SDL-Net. No output stride, therefore no quantisation error from upsampling a
coarse heatmap back to input resolution.

The common pose-estimation convention (stride 4, e.g. 128×128 heatmaps) would be ~16× cheaper and
is a reasonable fallback if memory or throughput demand it — but it introduces a quantisation floor
of ±2 px, and corner precision is the whole point here. Full-res first.

### σ = 8 px at 512

Derivation: standard practice in keypoint work is σ ≈ 2 px on stride-4 heatmaps, i.e. **8 px in
input-image space**. Applied at full resolution, that is σ = 8.

`[ASM-05]` — **σ = 8 is a starting point, not a tuned value.** Sweep σ ∈ {4, 8, 12} in Phase 06 and
report the effect. The trade is standard: small σ gives precise peaks but a sparse, hard-to-optimise
target; large σ trains easily but blurs the peak location.

### Render the Gaussian in a local window — mandatory, not an optimisation

Evaluate the Gaussian only within ±3σ of the corner (a 49×49 patch at σ=8) and paste it into a zero
array. Beyond 3σ the value is under 0.02% of peak, so the result is numerically indistinguishable
from a full-frame render at roughly **1/100th the cost** (~2.4k evaluations vs ~262k, per corner,
per sample).

This matters concretely: four full-frame 512² Gaussians per sample on Colab's ~2 vCPUs is a
significant fraction of the CPU budget the generator does not have (ADR-003).

**Edge handling:** when a corner is near the frame border the window is clipped. Clip the patch, do
not shift it — shifting moves the peak and corrupts the label. Peak value stays exactly 1.0 at the
corner pixel.

### Target range `[0,1]`, peak exactly 1.0. Output through sigmoid.

Unnormalised Gaussians (peak = `1/(2πσ²)`) produce tiny target values and a vanishing loss signal.
Normalise so the peak is 1.

### Loss: MSE first, foreground-weighted MSE as a pre-approved fallback

**Start with plain MSE.** It is the spec's default reading of "a pixel-wise loss", and it is what
the analysed baseline used to reach 96% synthetic success — so it demonstrably trains.

**The known risk, stated so it is recognised rather than debugged from scratch:** at σ=8 on a 512²
map, the Gaussian's effective support (~π(3σ)² ≈ 1,800 px) is about **0.7% of the 262,144 pixels**.
The loss is dominated by background. The literature is clear that MSE is a poor choice for heatmap
regression for exactly this reason, and that this is what Adaptive Wing Loss was designed to fix.

**Failure signature:** predicted heatmaps collapse to near-zero everywhere, loss drops fast then
plateaus at a low value, argmax returns noise. If you see that, it is not a bug — it is this
imbalance.

**Pre-approved response** (no escalation needed; log it as an experiment): switch to
foreground-weighted MSE — weight each pixel by `1 + w·target`, with `w` ≈ 10–50, so pixels near the
peak count more. Cheap, one line, keeps the "pixel-wise loss" framing of `[REQ-30]`.

**Adaptive Wing Loss** (Wang et al., ICCV 2019) is the literature's stronger answer and reports
real gains. It is *not* adopted by default: it adds hyperparameters and complexity for a network
that may well train fine without it. Escalate before adopting it — `05-skills/scope-guard.md`.

### Coordinate extraction: argmax + local soft-argmax refinement

Two-stage:
1. **argmax** over each channel → integer peak `(x̂, ŷ)`. Robust; unaffected by other modes.
2. **Local soft-argmax** in a small window (e.g. 11×11) centred on that peak → sub-pixel
   coordinate, as the intensity-weighted centroid of the window.

This is better than either extreme. Plain argmax is limited to integer precision. *Global*
soft-argmax over the whole heatmap is fragile: it computes an expectation over the entire map, so a
spurious secondary peak — precisely what a cluttered background produces — drags the estimate
towards the midpoint between two blobs. The local window keeps sub-pixel accuracy while remaining
robust to multimodality.

The research report's "centroid of the predicted heatmap contour" is the same instinct; the local
soft-argmax is the cleaner formulation of it.

**Also record the peak value** (max activation per channel) as a confidence score. SDL-Net uses it
the same way. Free, and useful for failure analysis: low confidence on a bad prediction is a very
different story from high confidence on a bad prediction.

### For the differentiable bonus (`[REQ-42]`)

The local-window soft-argmax is differentiable with respect to the heatmap values, so it can carry
gradients from the enhancement loss back into the corner detector. The argmax that *selects* the
window is not differentiable — treat the window position as fixed per forward pass (a
straight-through arrangement). Document this when implementing Phase 09; it is a standard and
defensible construction, but it must be stated rather than glossed.

## Consequences

**Good.** Full precision, no quantisation floor, cheap to render, robust extraction, free
confidence signal, and a differentiable path to the bonus.

**Costs.** Four 512² float targets per sample (4 MB fp32). Consider generating targets as float16 or
constructing them on the GPU if host memory pressure appears on Colab.

**Open items to resolve in Phase 06:** the σ sweep (`[ASM-05]`), and whether plain MSE suffices or
the weighted variant is needed. Both are logged in `state/assumptions.md`.

## Alternatives considered

- **Stride-4 (128×128) heatmaps.** ~16× cheaper, standard in pose estimation. Rejected as default
  for the ±2 px quantisation floor; kept as the fallback if throughput or memory demands it.
- **Global soft-argmax only.** Fully differentiable end-to-end and elegant, but fragile to
  secondary peaks from clutter — the exact condition this project's hard negatives create.
- **Direct offset regression per pixel (CenterNet-style).** Fixes quantisation at stride, but adds
  a second head and more machinery than a full-resolution heatmap needs. `[CON-10]`.
- **A single 4-corner heatmap with corner association.** Needed for multi-document detection; this
  project has exactly one document per image, and four channels give ordering for free.
