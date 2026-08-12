# Synthetic Generator Specification

The most important component in the project. Its realism determines the ceiling on everything
downstream (`02-research/sim2real-playbook.md`).

**Governing requirements:** `[REQ-07]`, `[REQ-08]`, `[REQ-09]`, `[REQ-33]`, `[REQ-34]`, `[REQ-35]`,
`[REQ-36]`, `[REQ-37]` · **Constraints:** `[CON-03]` (OpenCV+NumPy only), `[CON-05]` (no flips),
`[CON-08]` (no corner coords into the enhancement net), `[CON-09]` (order is fixed)

> `[REQ-09]` is the requirement that shapes this whole file: "The enhancement network operates on
> the **rectified crop**, not on the raw photo. Its input is the degraded document warped back to a
> flat rectangle; its target is the original clean scan." That is why one generator call must emit
> **two different framings** of the same sample — the composite for the corner detector, and the
> rectified crop for the enhancement network.

---

## 1. What one call produces

A single generator call takes a clean scan and a background and produces **everything both tasks
need**, which is `[REQ-08]`: "the label generator and the data generator are the same function."

```
generate(clean_scan, background, rng) -> sample

sample:
  composite       (512, 512, 3) uint8 BGR   degraded photo-like image  ─┐ corner detector
  corners         (4, 2) float32 (x,y) abs px in composite frame       ─┘ input + label

  enhance_input   (512, 512, 3) uint8 BGR   composite rectified back   ─┐ enhancement
  enhance_target  (512, 512, 3) uint8 BGR   clean scan, same framing   ─┘ input + target

  H               (3, 3) float64            scan-space -> composite-space
  params          dict                      every sampled value, for logging/QA
```

`params` is not optional. It is what makes the coverage plot
(`02-research/sim2real-playbook.md` §2) and every debugging session possible.

---

## 2. Pipeline order — fixed by `[REQ-34]`, do not reorder

```
  clean scan ──┐
               │  ① perspective warp onto background  ─────► composite + corners + H
  background ──┘
                  ② downscale ×[2,4] then upscale back
                  ③ brightness · contrast · colour cast
                  ④ illumination gradient × + soft shadows
                  ⑤ gaussian blur → gaussian noise
                  ⑥ JPEG encode/decode, quality [30,80]
                                │
                                ├──────────────────────────► composite  (corner detector input)
                                │
                                └── warp back with H⁻¹ ────► enhance_input
                                                             (enhancement input)

  clean scan ──── resize to 512×512 ───────────────────────► enhance_target
```

The order is physically motivated: geometry is established by the camera, then the whole scene is
degraded, then the file is compressed. Steps 2–6 apply to the **composite**, which is why the
enhancement input inherits realistic degradation *through* the rectification.

---

## 3. Alignment — the rule that pixel losses depend on

`[REQ-35]`: photometric degradation goes on the **input only**, never the target. The geometric
warp must be inverted **exactly**, using the same `H` that produced it.

- `enhance_target` is the clean scan resized to 512×512. It never touches steps 2–6.
- `enhance_input` is `warpPerspective(degraded_composite, inv(H_full), (512,512))`, where `H_full`
  is the composed transform from *clean-scan pixel space* to *composite pixel space*.
- Use `cv2.invert` or `np.linalg.inv` on the composed matrix. **Do not** re-derive an inverse
  homography from the corner points — floating-point differences will misalign by a pixel or two,
  and `[REQ-35]` warns exactly about this: "If the input and target drift out of alignment by even
  a few pixels, pixel-wise losses will punish the model for errors it did not make."

**Verification gate** (Phase 02): run the generator with a `photometrics_off` debug flag so only
step ① applies, then round-trip. `PSNR(round_trip, target)` must exceed **30 dB**. It will not be
infinite — resampling twice loses real information — but anything much below 30 dB means the
homography composition is wrong.

---

## 4. Aspect-ratio policy — one rule, applied everywhere

**Everything is stretched to a square 512×512. No letterboxing.**

Applies identically to: `enhance_target` (clean scan → square), `enhance_input` (rectified crop →
square), `composite` (canvas is square), real-photo preparation, and both inference pipelines.

Rationale: it is the simplest reading of spec §2.2 step 3, and normalised coordinates (`x/W`, `y/H`)
handle non-uniform scaling correctly **provided the same W and H are used in both directions**.

> ⚠️ The failure mode is *mixing* policies — a letterboxed generator with a stretched inference
> path, or vice versa. That produces a small, systematic corner error that looks like model
> inaccuracy. Write a test that round-trips a known coordinate through the resize path in both
> directions.

At inference the aspect ratio of the *output* is restored when the enhanced image is resized back
to the original dimensions (`[REQ-29]`).

---

## 5. Step ① — perspective warp onto a background

The step that most determines sim2real success. `02-research/baseline-failure-analysis.md` traces
the 96%→0% collapse to fixed parameters here.

**Procedure**
1. Load a background, resize/crop to 512×512.
2. `[REC]` Optionally composite 0–2 **distractor quadrilaterals** (ADR-004 §2) — dimmer,
   paper-like, possibly partially outside the frame. **Ablation, not baseline**: build without
   them first.
3. Sample four target corners in the 512×512 canvas (below).
4. `H = cv2.getPerspectiveTransform(scan_corners, target_corners)` — both `(4,2) float32`,
   both in TL,TR,BR,BL order (`00-project/conventions.md` §1).
5. `cv2.warpPerspective` the scan onto a copy of the background, with a mask so only the page
   region is written.
6. `[REC]` **Edge shadow.** Blur the page mask and darken the background just outside the page
   boundary. Real pages cast a thin contact shadow; without it the composite has a tell-tale
   razor-sharp seam. Cheap, pure OpenCV, and one of the highest-value realism wins
   (`sim2real-playbook.md` §6).
7. Record `target_corners` as the label.

**Corner sampling — kill the spatial priors**

Sample **shape first, then place it**, rather than jittering four points independently. Independent
jitter produces mostly near-rectangles and occasionally degenerate quads; shape-then-place gives
direct control over the distribution you actually care about.

| Parameter | Suggested range | Why |
|---|---|---|
| Page area fraction | **0.15 – 0.95** of canvas | Baseline's fixed 15% margin was the failure. Wide scale is essential. |
| In-plane rotation | **±25°** (calibrate, then widen) | Real photos are roughly upright; `[CON-05]` reasoning excludes upside-down. |
| Perspective severity | **0.0 – 0.35** (normalised corner displacement) | Baseline used a fixed 0.08. Must span near-fronto-parallel to strong. |
| Edge margin | **≥ 0** — corners may touch the border | Removes the "always inset" prior. |
| Aspect jitter | **±15%** on the page's own aspect | Simulates non-A4 pages and mild framing effects. |

**Hard rules**
- All four corners **inside the frame**. Truncated pages are out of scope (ADR-004 §4) — a corner
  outside the canvas has no heatmap peak. Note it as a limitation under `[REQ-48]`.
- **Reject degenerate quads.** After sampling, verify convexity, verify the corners are still in
  TL,TR,BR,BL order (cross-product sign test), and verify no interior angle is below ~20°. Resample
  on failure. A self-intersecting quad produces a folded warp and a nonsense training pair.
- Every parameter redrawn per sample (`[REQ-36]`).

**Calibration** (ADR-004 §3): after Phase 01, measure these same statistics on the real photos and
reset the ranges to **cover the observed distribution, then widen ~1.5–2×**. The table above is the
provisional starting point.

---

## 6. Step ② — resolution loss

Downscale by a random factor in `[2, 4]`, then upscale back to 512×512.

- Randomise the interpolation, not just the factor: `INTER_AREA`/`INTER_LINEAR` down,
  `INTER_NEAREST`/`INTER_LINEAR`/`INTER_CUBIC` up. Different phones and different distances produce
  different resampling signatures.
- Sample the factor continuously, not from `{2,3,4}`.

Simulates photographing from a distance and limited effective sensor resolution.

---

## 7. Step ③ — brightness, contrast, colour cast

`out = clip(α · img + β)` with a per-channel gain for the cast.

| Parameter | Suggested range |
|---|---|
| Contrast α | 0.7 – 1.3 |
| Brightness β | −40 – +40 (on 0–255) |
| Warm/cool cast | R and B channel gains in 0.9 – 1.1, **anti-correlated** |

Anti-correlating the R and B gains is what makes it a *colour temperature* shift rather than
generic channel noise. **In BGR, index 0 is blue and index 2 is red** — a warm cast raises index 2
and lowers index 0 (`00-project/conventions.md` §3).

Work in float, clip once at the end. Clipping between sub-steps quantises repeatedly and loses
detail.

---

## 8. Step ④ — illumination gradient and shadows

**The most characteristic defect of real document photos, and the one the enhancement network
mainly exists to fix.** Give this step the most care.

**Illumination gradient.** Build a smooth multiplicative mask over the full canvas and multiply.
- Randomise: direction (any angle 0–360°), steepness, and whether it is linear or radial
  (vignette-like).
- Suggested mask range 0.55 – 1.15. Values above 1 matter — real lighting produces bright hotspots
  as well as falloff.

**Soft shadows.** Composite 0–3 blurred polygons at reduced intensity.
- Randomise per shadow: number of vertices (3–7), size, position (may overlap the page edge or
  extend off-canvas), rotation, **blur kernel size** (this is the penumbra — vary it widely; hard
  and soft shadows look very different), and opacity (roughly 0.15–0.55).
- **Randomise presence.** A meaningful fraction of samples should have no shadow at all — otherwise
  the network learns that every document has one and will invent shadow removal where there is
  nothing to remove.

`[REQ-36]` bites hardest here: "A model trained on one shadow direction learns that shadow
direction, not shadows."

---

## 9. Step ⑤ — blur then noise

**In that order** (`[CON-09]`). Physically: optical blur happens in the lens, sensor noise is added
afterwards. Blurring after adding noise would smooth the noise and produce an unrealistic clean-ish
result.

- Gaussian blur: kernel 3–9 px, σ sampled continuously.
- `[REC]` Slight **motion blur** as an occasional alternative — a small directional kernel via
  `cv2.filter2D`, random angle. The spec names "Gaussian or slight motion blur" (§4.1). Camera shake
  is directional and looks different from defocus; including it is cheap realism.
- Gaussian noise: σ roughly 2–12 on the 0–255 scale.
- `[REC]` Real sensor noise is stronger in dark regions. Scaling noise σ by local darkness is a
  small, cheap realism gain — and `sim2real-playbook.md` §6 lists uniform noise as a common
  synthetic giveaway.

---

## 10. Step ⑥ — JPEG re-encode

`cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])` then `cv2.imdecode`, with `q` sampled
uniformly in **[30, 80]** (`[REQ-34]`).

Last step, always. Introduces the 8×8 block artifacts and chroma subsampling that every phone photo
carries.

> **Frozen evaluation sets must be written as PNG**, never JPEG — saving as JPEG would apply a
> second, uncontrolled compression on top of this step (ADR-003).

---

## 11. Performance — this runs on ~2 vCPUs

ADR-003 identifies CPU throughput as the main risk. Design accordingly from the start:

- **Cache decoded, pre-resized scans and backgrounds in RAM** at worker startup. Re-decoding a JPEG
  every `__getitem__` is the single biggest avoidable cost. This does not violate `[REQ-11]` —
  compositing is still fresh per call.
- **Work at 512×512 throughout.** Do not composite at 1024 and downscale.
- Keep intermediates as `uint8` where possible; use `float32` only inside steps 3–4 where the maths
  needs it.
- Avoid per-pixel Python loops entirely. Everything here is expressible as vectorised NumPy or a
  single OpenCV call.
- **Heatmap targets: render in a ±3σ window and paste** (ADR-008). Never evaluate a Gaussian over
  the full 512² canvas — it is ~100× more work for a numerically identical result.

**Phase 02 gate:** benchmark samples/s at 1, 2 and 4 workers and record the numbers.

---

## 12. Verification — `[REQ-37]`, and a Phase 02 gate

Full procedure in `05-skills/synthetic-data-qa.md`. The gate items:

1. **Round-trip alignment**: photometrics off → `PSNR(round_trip, target) > 30 dB`.
2. **Corner overlay**: corners drawn on the composite land on the page corners, colour-coded per
   `00-project/conventions.md` §8. Verify on ≥20 samples including extreme geometry.
3. **Order check**: `H` applied to the scan's own corners reproduces the recorded corners to within
   1e-3.
4. **No degenerate quads** across 1000 samples: all convex, all correctly ordered.
5. **Stranger test** (`[REQ-37]`): shuffled grid of synthetic and real samples; giveaways listed in
   `sim2real-playbook.md` §6.
6. **Readability**: in the worst 10 of 100 samples sorted by severity, the text is still readable by
   eye — "be cautious of excessive degradation" (`[REQ-37]`).
7. **Parameter coverage**: histograms of every sampled parameter show the intended ranges, with no
   accidental constants and no clipping against a bound.
8. **Throughput**: samples/s recorded at 1/2/4 workers.

---

## 13. Configuration

Every range above is a config value, never a literal (`00-project/conventions.md` §9). Shape:

```yaml
generator:
  canvas: 512
  geometry:
    area_fraction:        [0.15, 0.95]
    rotation_deg:         [-25, 25]
    perspective_strength: [0.0, 0.35]
    aspect_jitter:        [-0.15, 0.15]
    min_interior_angle_deg: 20
  resolution_loss:
    scale_factor:         [2.0, 4.0]
  photometric:
    contrast:             [0.7, 1.3]
    brightness:           [-40, 40]
    channel_gain:         [0.9, 1.1]
  illumination:
    gradient_range:       [0.55, 1.15]
    shadow_count:         [0, 3]
    shadow_opacity:       [0.15, 0.55]
    shadow_blur:          [15, 91]
    shadow_probability:   0.7
  sensor:
    blur_kernel:          [3, 9]
    motion_blur_prob:     0.25
    noise_sigma:          [2, 12]
  compression:
    jpeg_quality:         [30, 80]
  hard_negatives:
    distractor_count:     [0, 0]     # ablation: raise to [0,2] in Phase 06
  debug:
    photometrics_off:     false      # for the round-trip alignment gate
```

`[REQ-43]` requires being able to "add a new degradation" on request at the presentation — a
config-driven pipeline makes that a live demonstration rather than a code hunt.
