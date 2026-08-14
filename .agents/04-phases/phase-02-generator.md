# Phase 02 — Synthetic Generator

## Objective

Build the OpenCV degradation pipeline that produces every training sample and every label for both
tasks. **This is the highest-leverage phase in the project.** It costs no GPU time and it sets the
ceiling on everything after it.

## Prerequisites

Phase 00 gate passed. Backgrounds available (DTD at minimum; self-shot preferred). Can start on
stand-in document images if the provided scans are delayed (`03-spec/data-contract.md` §6).
Calibration ranges from Phase 01 land here when ready — start with the provisional table.

## Requirements in force

`[REQ-07]`, `[REQ-08]`, `[REQ-33]`, `[REQ-34]`, `[REQ-35]`, `[REQ-36]`, `[REQ-37]` ·
`[CON-03]` **OpenCV + NumPy only** · `[CON-05]` no flips · `[CON-09]` order fixed ·
ADR-003, ADR-004

## Read before starting — non-optional

- `02-research/baseline-failure-analysis.md` — how a 96% synthetic score became 0% on real photos
- `02-research/sim2real-playbook.md` — the strategy that prevents it
- `03-spec/synthetic-generator-spec.md` — the full specification

---

## Tasks

### A. Geometry
1. `src/geometry/homography.py`: corner ordering utilities, convexity and ordering validation
   (cross-product sign), and the coordinate-scaling round-trip.
2. Corner sampling per `synthetic-generator-spec.md` §5: **sample the quad's shape, then place it**
   — not four independently jittered points. Independent jitter yields mostly near-rectangles and
   occasional degenerate quads, and gives no control over the distribution that matters.
3. Rejection loop: resample if the quad is non-convex, mis-ordered, or has an interior angle below
   ~20°.
4. Compose `H` from scan space → composite space; keep the **exact** matrix for inversion
   (`[REQ-35]`).

### B. The six degradations — `[REQ-34]` order, `[CON-09]`
5. ① Warp onto background, with a page mask. `[REC]` add the **edge contact shadow** — cheap, and
   one of the biggest realism wins (`sim2real-playbook.md` §6).
6. ② Downscale ×[2,4] → upscale. Randomise the interpolation method too, not just the factor.
7. ③ Brightness / contrast / colour cast. Work in float, **clip once** at the end.
   Remember: **BGR — index 2 is red, index 0 is blue.**
8. ④ Illumination gradient (linear or radial, any direction) × soft shadows (0–3 blurred polygons,
   randomised vertices/size/position/**blur**/opacity, and randomised **presence**).
9. ⑤ Gaussian blur → Gaussian noise, **in that order**. `[REC]` occasional motion blur;
   `[REC]` darkness-scaled noise.
10. ⑥ JPEG encode/decode, quality ∈ [30, 80].
11. **Every parameter sampled per call** (`[REQ-36]`) and recorded in `params`.

### C. Dual output — `[REQ-08]`
12. Return `composite` + `corners` (corner task) and `enhance_input` + `enhance_target`
    (enhancement task) from one call.
13. `enhance_input = warpPerspective(degraded_composite, inv(H_full))`.
    **Invert the matrix — do not re-derive a homography from the corner points** (`[REQ-35]`);
    floating-point differences will misalign by a pixel or two and pixel losses will punish the
    model for errors it did not make.
14. `enhance_target` = clean scan resized to 512×512, **never photometrically degraded**.
15. `photometrics_off` debug flag for the alignment gate.

### D. Heatmap targets
16. Gaussian rendering in a **±3σ window, pasted** — never full-frame (ADR-008). ~100× cheaper,
    numerically identical.
17. Border handling: **clip the window, never shift it.** Shifting moves the peak and corrupts
    the label.

### E. Performance — ADR-003
18. Asset cache: decode and pre-resize scans and backgrounds into RAM at worker startup.
19. Benchmark: samples/s at 1, 2 and 4 workers. Record all three.

### F. Verification — `[REQ-37]`
20. Build the QA suite in `05-skills/synthetic-data-qa.md` and run every check.

---

## Gate

**Correctness**
- [ ] Round-trip alignment: `photometrics_off` → `PSNR(round_trip, target) > 30 dB`
- [ ] `H` applied to the scan's own corners reproduces the recorded corners to within 1e-3
- [ ] Corner overlay correct on ≥20 samples **including extreme geometry** (near-zero margin,
      maximum perspective, minimum and maximum scale)
- [ ] 1000 samples: all quads convex, all correctly ordered, none degenerate
- [ ] Coordinate scaling round-trips exactly through the resize path, both directions
- [ ] Degradation order matches `[REQ-34]` — verified by reading the code, not by intent

**Realism — `[REQ-37]`**
- [ ] **Stranger test:** shuffled grid of synthetic and real samples; giveaways from
      `sim2real-playbook.md` §6 checked for specifically
- [ ] **Readability:** in the worst 10 of 100 samples sorted by severity, text is still readable by
      eye ("be cautious of excessive degradation")
- [ ] Parameter histograms show the intended ranges — **no accidental constants**, no clipping
      against a bound
- [ ] Coverage plot: real-photo statistics sit **inside** the synthetic distribution, not at its
      edge (once Phase 01 data exists)

**Constraints**
- [ ] `grep` confirms no `albumentations` / `imgaug` / `kornia.augmentation` /
      `torchvision.transforms` in the generator (`[CON-03]`)
- [ ] No flips anywhere (`[CON-05]`)

**Performance**
- [ ] samples/s recorded at 1, 2, 4 workers
- [ ] Asset cache working; `persistent_workers` keeps it warm

---

## Failure modes

**Re-deriving the inverse homography from corner points.** The most likely silent bug in this phase.
It looks right, misaligns by 1–2 px, and the enhancement network then spends its capacity learning
a systematic sub-pixel shift. **Invert the matrix.**

**Ranges too narrow.** The baseline's exact failure. Symptom: everything looks fine, synthetic
scores are excellent, real photos collapse. Prevented by the parameter histograms and the coverage
plot — run them.

**Ranges too wide.** The opposite failure, and `[REQ-37]` warns about it: text destroyed, nothing to
recover, the model learns to hallucinate. The worst-10 readability check catches it.

**Any fixed parameter.** `[REQ-36]` — one fixed shadow direction teaches that direction, not
shadows. The histograms catch this: a fixed parameter shows as a spike.

**Photometrics leaking onto the target.** Breaks `[REQ-35]` and makes the task partly trivial —
metrics look great, the model has learned nothing useful. Assert the target is untouched.

**BGR/RGB confusion in the colour cast.** Warm casts come out cool. Obvious once you look at the
images, invisible in the numbers.

**Naive full-frame Gaussian rendering.** Not wrong, just ~100× too slow — and on Colab's 2 vCPUs
that alone can starve the GPU.

**Degenerate quads.** A self-intersecting quad produces a folded warp and a nonsense pair. Without
the rejection loop these appear rarely and poison training invisibly.

---

## Skills

- `05-skills/synthetic-data-qa.md` — **mandatory in this phase**
- `05-skills/scope-guard.md` — the temptation here is to add degradations beyond the six specified

---

## Deliverables

| Artifact | Location |
|---|---|
| Degradation pipeline | `src/data/generator.py` |
| Geometry utilities | `src/geometry/homography.py` |
| Generator config with calibrated ranges | `configs/base.yaml` |
| QA suite | `tests/test_generator.py` |
| Sanity panel figure | `outputs/figures/p02_samples.png` |
| Round-trip alignment proof | `outputs/figures/p02_roundtrip.png` |
| Stranger-test figure | `outputs/figures/p02_stranger.png` |
| Parameter histograms | `outputs/figures/p02_params.png` |
| Coverage plot (real vs synthetic) | `outputs/figures/p02_coverage.png` |
| Throughput numbers | `state/discoveries.md` |
