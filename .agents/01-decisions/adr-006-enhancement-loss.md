# ADR-006 — Enhancement Loss Function and the Ablation Set

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** High

## Context

Spec §3.2 poses this as an open question and then hints heavily:

> "A standard pixel-wise loss like Mean Squared Error is known to produce **blurry outputs** in
> image restoration — and blur is precisely the enemy when the goal is readable text. Do you have
> any idea?"
> *Hint:* "Investigate the L1 loss, the (MS-)SSIM loss, and losses computed on image *gradients*
> (e.g., L1 between Sobel edge maps). Combinations of these are a well-known recipe in the image
> restoration literature. Text legibility lives in the edges."

Combined with `[REQ-45]` — "include comparisons between different methods (**loss functions**, and
regression vs. heatmap)" — **comparing loss functions is a graded deliverable**, not an optional
exploration. The deliverable is the comparison; the winning loss is a by-product.

Why L2 blurs: MSE is minimised by the conditional *mean* of the plausible outputs. Where the
network is uncertain about a stroke edge's exact position, averaging over candidate positions
costs less than committing to the wrong one. The result is systematically smooth. L1 is minimised
by the conditional *median*, which does not reward hedging in the same way, so edges stay sharper.

## Decision

### The ablation set — four variants, all at 512 (ADR-002)

| ID | Loss | Role |
|---|---|---|
| **L-A** | `MSE` | The spec's named straw man. Establishes the blur baseline the report needs. |
| **L-B** | `L1` | Isolates the pixel-loss change alone. |
| **L-C** | `α·(1−MS-SSIM) + (1−α)·L1`, **α = 0.84** | Expected winner. |
| **L-D** | `L-C + λ·SobelL1`, λ ≈ 0.1 | Tests whether an explicit edge term adds anything on top. |

Each is one experiment in `state/experiments.md`, identical in every other respect — same seed,
same architecture, same schedule, same frozen val/test. **One variable at a time**
(`05-skills/experiment-discipline.md`).

**Predicted outcome, recorded now so the comparison is honest:** L-C wins on SSIM and on visual
sharpness; L-A wins or ties on PSNR while looking clearly worse (PSNR is an L2-derived metric, so
it structurally favours the L2-trained model — say this in the report, it is exactly the kind of
insight `[REQ-45]` rewards). L-D is expected to be roughly neutral: the Sobel term may sharpen
strokes but can also amplify residual sensor noise, since the network cannot tell a text edge from
a noise edge. **If L-D loses, that is a result — report it.**

### On α = 0.84

From Zhao, Gallo, Frosio & Kautz, *Loss Functions for Image Restoration with Neural Networks*
(IEEE TCI 3(1):47–57, 2017; arXiv:1511.08861), which found a weighted MS-SSIM + L1 mixture best on
both distortion and perceptual metrics.

⚠️ **Verify the exact formulation against the paper before implementing.** The paper's mixed loss
applies a Gaussian weighting to the L1 term (derived from the coarsest MS-SSIM scale), i.e. roughly

```
L_mix = α · L_MS-SSIM + (1 − α) · G_σM · L1
```

Most public reimplementations drop the Gaussian weighting and use a plain L1 term. **The simplified
form is acceptable** — but read arXiv:1511.08861 §5, implement whichever you choose knowingly, and
state which in the report. Do not cite the paper for a formulation you did not implement.

`[ASM-04]`: α = 0.84 transfers to document images. It was tuned for natural-image restoration.
Documents are high-contrast and bimodal, so the optimum may differ. **Validate cheaply:** once L-C
is trained, run a short sweep at α ∈ {0.7, 0.84, 0.95}. If the ranking is flat, keep 0.84 and say
so; if not, use the best and report the sweep.

### MS-SSIM is a *loss*, so it must be differentiable and implemented by us

Per ADR-010, SSIM and MS-SSIM are hand-implemented and numerically validated against
`skimage.metrics.structural_similarity`. The loss is `1 − MS-SSIM` (MS-SSIM is a similarity in
`[0,1]`, higher is better).

**Constraints to respect:** MS-SSIM at 5 scales needs images ≥161 px — satisfied at 512.
Compute it in `[0,1]` space with `data_range=1.0`. Guard against NaN: the SSIM denominator can
vanish on perfectly flat patches (a blank margin of a document is exactly such a patch), so the
stabilising constants C1, C2 are load-bearing here, not decorative.

### Sobel term (L-D)

Apply a fixed Sobel kernel to prediction and target, take L1 between the gradient magnitudes.
Implemented as a fixed-weight `conv2d`, not a learned layer. Small weight (λ ≈ 0.1) — a large one
makes the network chase noise.

## Consequences

**Good.** Satisfies `[REQ-23]` and `[REQ-45]` directly. Four runs is a manageable matrix. The
predicted-outcome-in-advance framing makes the write-up stronger whichever way it lands.

**Costs.** Four full training runs at 512 (ADR-002 accepted this). MS-SSIM backward is noticeably
more expensive than L1 — expect L-C/L-D to be slower per epoch than L-A/L-B; that is not a bug.

**Trap.** Do not select the winner on the **test** split (`[CON-07]`). Select on validation, then
report the winner's test numbers once.

**Trap.** The four runs must be compared on the *same frozen val/test sets*. If the generator
changes mid-ablation, every earlier run is invalidated — finish the generator first (Phase 02 gate)
and freeze before Phase 04 starts.

## Alternatives considered

- **Perceptual/VGG or LPIPS loss.** The standard modern answer for sharpness. **Banned by
  `[CON-02]`** (pretrained weights). Worth one sentence in the report's "what we could not try".
- **Adversarial loss.** Would produce the sharpest text, but adds a discriminator and instability
  for a project graded on PSNR/SSIM/OCR. `[CON-10]`.
- **Charbonnier / smooth-L1.** A reasonable L1 variant, but adds a variant without adding a
  distinct *idea* to the comparison. Skip unless L1 shows optimisation trouble.
- **Pure MS-SSIM, no L1.** MS-SSIM is insensitive to uniform intensity shifts, so it can drift in
  overall brightness — exactly the wrong failure for a document that should be white. The L1 term
  is there to anchor global intensity, which is why the mixture exists.
- **Binarisation / segmentation-style loss.** Would suit binarisation benchmarks, but the target
  here is a colour scan (logos, figures), not a binary mask.
