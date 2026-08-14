# ADR-010 — Implement SSIM and MS-SSIM Ourselves

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** High
**Decided by:** the human.

## Context

SSIM appears twice in this project, in two different roles:

1. **As a required reported metric** — `[REQ-24]`, spec §3.3, alongside PSNR.
2. **As a differentiable loss term** — MS-SSIM in variants L-C and L-D of ADR-006.

Ready-made implementations exist and are good: `pytorch-msssim`, `torchmetrics`,
`piq`, and `skimage.metrics.structural_similarity` (the field's de-facto reference for
*reporting*, though it is not differentiable).

The constraint picture is genuinely ambiguous. `[CON-03]` bans third-party libraries for the
**transformations** (spec §4) — a loss function is not a transformation. `[CON-01]`/`[CON-02]` ban
pre-built **architectures and weights** — a loss function is neither. So importing MS-SSIM is
arguably permitted.

Against that: `[REQ-43]` requires being "prepared to explain and modify any part of the code if
asked."

## Decision

**Implement SSIM and MS-SSIM by hand, and validate them numerically against `skimage`.**

Reasoning:

- **The grey area resolves in favour of writing it.** The spec's posture throughout is
  build-it-yourself; the assignment is graded partly on demonstrated understanding. When a
  restriction is ambiguous, the version that is unambiguously safe costs a couple of hours here.
- **Presentation risk is the real driver.** "Explain the MS-SSIM scale weights" or "change the loss
  to weight the finest scale more" is a plausible live question. Reading someone else's library
  under that pressure is a bad position; modifying your own is a 30-second demonstration.
- **It is genuinely small.** SSIM is a Gaussian-windowed computation over local means, variances and
  covariance. MS-SSIM is SSIM at five scales with average-pool downsampling and fixed weights. Both
  fit comfortably in ~80 lines.
- **The main objection — subtle-bug risk — is fully addressable**, and the validation test below
  addresses it.

### Mandatory validation test

A unit test, written in the same session as the implementation, asserting agreement with
`skimage.metrics.structural_similarity` to within **1e-4** on:
- random noise images,
- a real document sample,
- identical images (must give exactly 1.0),
- a pair with a known constant offset,
- a flat/uniform patch (the degenerate case where the denominator nearly vanishes — a blank document
  margin is exactly this, so it is not a hypothetical).

**This test is a Phase 04 gate item.** An unvalidated SSIM implementation silently corrupts every
number in the results table and the loss that trains the winning model. It is not optional and it
is not "test it later".

### Pinned parameters

Match `skimage` defaults so the comparison is meaningful, and pin them in config:

| Parameter | Value |
|---|---|
| Gaussian window | 11×11 |
| σ | 1.5 |
| K1, K2 | 0.01, 0.03 |
| `data_range` | 1.0 (ADR-009: metrics in `[0,1]`) |
| MS-SSIM scales | 5 |
| MS-SSIM scale weights | `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]` (Wang et al. 2003) |

Note: `skimage`'s `structural_similarity` defaults to a **uniform** 7×7 window unless
`gaussian_weights=True` is passed. Set `gaussian_weights=True, sigma=1.5, use_sample_covariance=False`
in the test to compare like with like — otherwise the test will fail for a correct implementation
and cost an afternoon.

### Reporting

Report metrics with **our** implementation, so the reported SSIM and the SSIM the loss optimised are
the same quantity. `skimage` remains in the test suite as the reference. If they ever disagree
beyond tolerance, that is a bug to fix, not a discrepancy to report.

### Constraints to respect

- **MS-SSIM needs ≥161 px** at 5 scales (`(11−1)·2⁴ + 1`). 512 is fine; if anything ever runs at a
  smaller size, reduce the number of scales rather than letting it error.
- **Numerical stability:** C1 and C2 exist to keep the denominator away from zero on flat patches.
  Do not "simplify" them away.
- Compute in `float32`. In AMP, cast the loss computation to `float32` explicitly — MS-SSIM's
  products and divisions are not fp16-friendly and will produce NaN.

## Consequences

**Good.** Zero ambiguity about the constraints. Full ability to explain and modify at the
presentation. Loss and metric are provably the same function. Validation test doubles as
documentation of intent.

**Costs.** A few hours, plus the validation test. Likely slower than a tuned library implementation
— irrelevant relative to the network's own cost, but if MS-SSIM backward becomes a measured
bottleneck, that is a legitimate reason to revisit.

**Risk.** A subtle error that the validation test does not cover. Mitigation: the flat-patch and
identical-image cases above are chosen to cover the usual culprits (window normalisation,
`use_sample_covariance`, denominator stabilisation).

## Alternatives considered

- **`pytorch-msssim` / `torchmetrics`.** Battle-tested, zero bug risk, saves a few hours.
  Rejected on the grey-area and presentation-risk grounds above.
- **Own loss, `skimage` for reporting.** Attractive — `skimage` is the reference implementation.
  Rejected because the reported metric would then differ from the optimised one, and any
  disagreement becomes an awkward footnote rather than a bug to fix.
