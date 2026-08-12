# ADR-009 — Normalisation, Value Ranges, and Tensor Conventions

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Medium

## Context

Spec §2.2 gives two related instructions:

- Step 4: "Scale your image's pixel values (e.g., divide by 255.0 to get values in the range
  [0, 1]). **Important:** Normalize the corner coordinates as well (divide by image width and
  height) so that they live in [0, 1]." (`[REQ-13]`)
- Step 5: "For PyTorch, apply `transforms.Normalize(mean, std)` after resizing."

Step 5 raises a question the spec does not resolve: for an **image-to-image** task, does mean/std
standardisation apply to the target as well as the input? If it does, PSNR and SSIM must be
computed after un-standardising, and the model's output range becomes unbounded. Get this wrong and
the metrics are silently on the wrong scale — a whole class of confusing results.

There is a second issue: `transforms.Normalize` is conventionally used with **ImageNet** mean/std,
which exist to match a pretrained network's input statistics. `[CON-02]` forbids pretrained
weights, so those constants have no meaning here.

## Decision

### Asymmetric normalisation: standardise the input, leave the target in `[0,1]`

| Tensor | Range | Rationale |
|---|---|---|
| **Enhancement input** | standardised: `(x/255 − mean) / std` | Satisfies spec §2.2 step 5; conditions optimisation |
| **Enhancement target** | `[0,1]` (`/255` only) | Keeps the output space bounded and metric-ready |
| **Enhancement output** | `[0,1]` via `sigmoid` | Matches the target; guarantees a valid image |
| **Corner-detector input** | standardised, same as above | Same rationale |
| **Approach A target/output** | `[0,1]` normalised coords, `sigmoid` output | `[REQ-13]` |
| **Approach B target/output** | `[0,1]` heatmaps, `sigmoid` output | ADR-008 |
| **All metrics** | `[0,1]`, `data_range=1.0` | PSNR/SSIM are only meaningful on a stated range |

**Why asymmetric.** The input is a *conditioning signal* — standardising it helps optimisation. The
output is an *image* — it must live in a valid, bounded pixel range so it can be saved, displayed
and scored. A sigmoid head onto a `[0,1]` target gives that for free and makes it impossible to
emit an invalid pixel. Standardising the target instead would require un-standardising before every
metric, every visualisation and every save — many places to get it wrong once.

This satisfies spec §2.2 step 5 as written (Normalize *is* applied after resizing, to the input)
while keeping the loss and metrics on a sane scale.

### Mean/std are computed from this project's data, not ImageNet

Compute per-channel mean and std **once**, in Phase 03, over the **training split only**, from
generated degraded inputs (typically a few thousand samples — the estimate converges quickly).
Store them in the config; never recompute per run, or runs become incomparable.

Using validation or test data to compute normalisation statistics is leakage. Training split only.

`[REC]`: if the computed values land near `mean ≈ 0.5, std ≈ 0.25` per channel, using those round
numbers is fine and slightly more robust to a regenerated dataset. Log whichever you use.

### Sigmoid, not clamp

A `sigmoid` head is smooth and always differentiable. A `clamp` has zero gradient outside `[0,1]`,
so any pixel that saturates stops learning — a real risk on documents, whose targets are dominated
by near-white background pixels sitting exactly at the top of the range.

**Watch for the related issue:** sigmoid saturates slowly near 1.0, so pure-white regions can be
persistently slightly grey. If the output has a systematic dull cast, that is the cause. Options:
scale the target slightly into `[0.02, 0.98]`, or switch to a linear head with an L1 loss. Log the
choice; do not silently change it mid-ablation.

### Tensor layout

`NCHW`, `float32`, RGB. Conversion from OpenCV's `HWC`/BGR/`uint8` happens exactly once, at the
Dataset boundary. Full table in `00-project/conventions.md` §3.

### Corner coordinate normalisation

`[REQ-13]`: divide by width and height so coordinates are resolution-independent. Since everything
runs at 512×512 (ADR-002), normalisation is division by 512 — but **keep it parameterised**, never
hardcoded, so mapping back to the *original* photo resolution in `[REQ-32]` uses the right divisor.

**Aspect-ratio warning.** Real photos are not square. Resizing a 4:3 photo to 512×512 stretches it
non-uniformly, and `x/W`, `y/H` handle that correctly *provided the same W and H are used in both
directions of the transform*. The failure mode is mixing a letterboxed pipeline with a stretched
one. Pick one policy, apply it in the generator, the real-photo preparation and both inference
pipelines, and state it in `03-spec/synthetic-generator-spec.md`. **Inconsistency here is a silent,
systematic corner error** — worth an explicit test.

## Consequences

**Good.** Metrics are always on a known scale. Outputs are always valid images. Satisfies the spec's
normalisation instruction without the awkwardness of un-standardising for every metric.

**Costs.** Two conventions to keep straight (standardised input, `[0,1]` target). Mitigated by
making it explicit in every docstring per `00-project/conventions.md` §10.

**Trap.** When visualising an input tensor, remember it is standardised — displaying it raw shows a
contrast-mangled image. Un-standardise for visualisation, or visualise the pre-tensor numpy array.

## Alternatives considered

- **Standardise both input and target.** Symmetric and defensible, but every metric and every
  saved image needs un-standardising. More places to make an invisible mistake.
- **`[0,1]` for both, no standardisation.** Simplest, and works fine in practice for image-to-image.
  Rejected only because spec §2.2 step 5 explicitly names `transforms.Normalize`.
- **`[-1,1]` with a `tanh` head.** Common in GAN-derived pipelines. No advantage here, and it adds
  a rescale before every metric.
- **ImageNet mean/std.** Meaningless without a pretrained network, and in tension with `[CON-02]`.
