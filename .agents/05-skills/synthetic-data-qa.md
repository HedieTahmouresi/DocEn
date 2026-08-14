# Skill: Synthetic Data QA

**Load when:** building or changing the data generator. **Mandatory in Phase 02.**

The generator is the only component whose bugs are invisible in the loss curve. A misaligned pair, a
frozen parameter, or a too-narrow range all produce healthy-looking training and a model that fails
on real photos. These checks are how you find that before spending GPU hours.

`[REQ-37]` requires verification. This is how.

---

## Tier 1 — Correctness. Must pass before anything else.

### 1.1 Round-trip alignment

Set `photometrics_off` so only the geometric warp applies, then round-trip the composite back
through `inv(H)` and compare to the target.

- **Expected: PSNR > 30 dB.** Not infinite — resampling twice loses real information.
- **Below 30 dB → the homography composition is wrong.**

The most likely cause, and the most common silent bug in this phase: **re-deriving the inverse
homography from the corner points instead of inverting the matrix.** It looks correct and misaligns
by 1–2 px, and the enhancement network then burns capacity learning a systematic sub-pixel shift.

### 1.2 Corners land on the page

Draw the recorded corners on the composite with the fixed colour code
(`00-project/conventions.md` §8): TL red, TR green, BR blue, BL yellow, edges drawn 0→1→2→3→0.

Check on **≥20 samples including the extremes** — near-zero margin, maximum perspective, minimum and
maximum scale. Easy geometry always looks fine; the extremes are where bugs live.

A **bowtie** shape means the ordering is wrong.

### 1.3 Corners are exactly `H` applied to the scan corners

`H @ scan_corners` must reproduce the recorded corners to within 1e-3. This catches the case where
the label and the actual warp have drifted apart.

### 1.4 No degenerate quads

Over 1000 samples: all convex, all correctly ordered by the cross-product sign test, no interior
angle below the configured floor.

A self-intersecting quad produces a folded warp and a nonsense training pair. Without a rejection
loop these appear rarely and poison training invisibly.

### 1.5 Coordinate scaling round-trips

Push a known coordinate through the resize path in **both** directions and back. `[REQ-12]`: "A
corner label that is not transformed together with its image is a wrong label."

Also verify one aspect-ratio policy is used everywhere — generator, real-photo prep, both inference
pipelines (`synthetic-generator-spec.md` §4). Mixing a letterboxed path with a stretched one gives a
small, systematic corner error that reads as model inaccuracy.

### 1.6 The target is untouched

Assert the enhancement target has received **no** photometric degradation (`[REQ-35]`). Leakage here
makes the task partly trivial: metrics look excellent and the model has learned nothing useful.

### 1.7 Constraint check

`grep` the generator for `albumentations`, `imgaug`, `kornia`, `torchvision.transforms`
(`[CON-03]`). And for any flip (`[CON-05]`).

---

## Tier 2 — Distribution. This is where sim2real is won or lost.

### 2.1 Parameter histograms

Generate 1000 samples, plot a histogram of **every** sampled parameter in `params`.

Look for:
- **A spike** → the parameter is effectively fixed. This is the baseline's failure
  (`[REQ-36]`: "Randomize *every* parameter within a range rather than fixing it").
- **Clipping against a bound** → the range is truncating; the effective distribution is not what the
  config says.
- **A gap** → a sampling bug.
- **Rejection-loop bias** → if the degeneracy rejection is firing often, it is silently reshaping the
  distribution. Log the rejection rate; if it exceeds a few percent, the sampling is wrong, not the
  rejection.

### 2.2 The coverage plot — the most valuable diagnostic in the project

Plot the **real photos' measured statistics** against the **same statistics sampled from the
generator**, for: page area fraction, in-plane rotation, perspective severity, edge margin,
brightness, contrast, blur, colour cast.

| Reading | Meaning |
|---|---|
| Real well inside synthetic | Good — real data is an easy interior case |
| Real at the edge | The model will be extrapolating. **Widen.** |
| Real **outside** synthetic | **This is your bug.** Found before wasting a training run |

Put this in the report. It is direct evidence for `[REQ-28]` and `[REQ-48]`.

### 2.3 The stranger test — `[REQ-37]`

Spec: "place a few generated samples next to the real test photos — if a stranger can instantly tell
which is which, your degradations are not yet realistic enough."

Build a shuffled grid and look at it honestly. Check specifically for the known giveaways:

- Backgrounds too **flat** — real surfaces have depth-of-field falloff and specular variation
- Shadow edges too **geometric** — real penumbras are wider and irregular
- The page too **evenly lit within itself** — real pages have subtle gradients even in good light
- A **razor-sharp seam** at the page boundary — real pages cast a thin contact shadow.
  `[REC]` Adding one is cheap, pure OpenCV, and one of the highest-value realism wins
- Noise **uniform across the image** — real sensor noise is stronger in shadows than highlights
- Colour cast applied **globally and identically** — real casts vary with local illumination

### 2.4 Readability — the counter-pressure

`[REQ-37]`: "Be cautious of excessive degradation, which might destroy the text entirely and leave
the model nothing to recover."

Generate 100 samples, sort by severity, look at the worst 10.
- Any unreadable → the upper bound is too high; the model is being trained on an impossible task and
  will learn to hallucinate.
- None noticeably harder than your hardest real photo → too low.

### 2.5 Ablate one degradation at a time

Generate a grid with each degradation applied **alone**. Two things this catches: a degradation that
silently does nothing (a bug), and a degradation that is far too strong once you see it in
isolation.

---

## Tier 3 — Performance

### 3.1 Throughput

Benchmark samples/s at 1, 2 and 4 workers. Record all three (`state/discoveries.md`).

Colab has ~2 vCPUs; the T4 consumes on the order of 40–80 samples/s at 512×512. If the generator
delivers well below that, the GPU will idle.

### 3.2 Profile before optimising

If throughput is low, profile a single `__getitem__` and find the actual cost. Usual order:
1. **Image decode** — solved by the RAM asset cache (ADR-003). Almost always the biggest win
2. **Full-frame Gaussian rendering** — ~100× slower than the windowed version (ADR-008)
3. `warpPerspective` — irreducible, it is the point
4. JPEG encode/decode — a few ms, irreducible, it is `[REQ-34]` step 6
5. Large blur kernels for shadows — can be significant; consider blurring at reduced resolution and
   upscaling the mask

Do not guess. The optimisation ladder is in ADR-003; apply it in order.

---

## When the generator changes after freezing

Any change that affects the evaluation distribution invalidates comparability with every earlier
run.

1. Bump `frozen_version`.
2. Regenerate the frozen val/test sets.
3. Record the regeneration as an event in `state/experiments.md`.
4. **Never mix `frozen_version`s in one table.**

This is expensive, which is the argument for getting Phase 02 right before Phase 04 starts.

---

## Quick checklist

```
Tier 1 — correctness
[ ] Round-trip PSNR > 30 dB (photometrics off)
[ ] Corner overlays correct on ≥20 samples incl. extremes
[ ] H @ scan_corners == recorded corners (1e-3)
[ ] 1000 samples: all convex, correctly ordered, non-degenerate
[ ] Coordinate scaling round-trips both directions
[ ] One aspect-ratio policy everywhere
[ ] Target free of photometric degradation
[ ] No banned imports; no flips

Tier 2 — distribution
[ ] Parameter histograms: no spikes, no clipping, no gaps
[ ] Rejection rate logged and low
[ ] Coverage plot: real inside synthetic
[ ] Stranger test passed; giveaways checked
[ ] Worst-10 readability check passed
[ ] Per-degradation ablation grid inspected

Tier 3 — performance
[ ] samples/s at 1, 2, 4 workers recorded
[ ] Asset cache working, persistent_workers on
```
