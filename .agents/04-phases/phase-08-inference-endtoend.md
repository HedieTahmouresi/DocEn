# Phase 08 — Inference Pipelines & the End-to-End Scanner

> **Rescoped 2026-08-14 (DEV-004).** This phase was "Bonus: chained scanner". It is now a
> **mandatory** phase, because the two inference pipelines it delivers are `[REQ-29]`,
> `[REQ-32]` and `[REQ-46]` — and `[REQ-49]` says the teaching staff run them on photos you
> have never seen. The chained scanner (`[REQ-40]`/`[REQ-41]`, the stated bonus) is glue
> over those two pipelines and comes nearly free, so it stays and is delivered here.
>
> Phase 09 (differentiable joint fine-tuning, the spec's flagged 🧩 Option) is **dropped**.
> ADR-012 already makes it conditional; its entry conditions are not met, and skipping it
> is the designed behaviour rather than a deviation.

## Objective

Make the project **runnable by someone else on inputs you have never seen.** Three entry
points, each accepting an arbitrary image and returning a result without crashing:

```
  rectified document image ──► enhance.py  ──► clean scan            [REQ-29]
  raw phone photo ───────────► corners.py ──► 4 corners + overlay    [REQ-32]
  raw phone photo ───────────► scanner.py ──► clean scan             [REQ-40] (bonus)
```

This is where the grade is actually decided. A pipeline that throws on an unusual aspect
ratio loses more marks than one that returns a mediocre result.

## Prerequisites

Phase 05 complete (enhancement evaluated, winner known). Phase 06 comparison re-run with a
fair Approach A (exp-011), so `[REQ-32]`'s "your better trained model" is an evidenced
choice rather than an assumed one. Phase 07 may run in parallel — the pipelines use the
un-regularised checkpoints unless the dropout variants win on validation.

## Requirements in force

`[REQ-29]` enhancement pipeline · `[REQ-32]` corner pipeline · `[REQ-40]` compose them ·
`[REQ-41]` evaluate the chain twice · `[REQ-43]` explain and modify any part live ·
`[REQ-46]` two pipelines, robust to variation · `[REQ-49]` graded on unseen photos ·
ADR-012

---

## Tasks

### A. Close the standardisation hole first — this is a live-demo bug

1. `preprocess_image_for_enhancement(config=None)` and
   `predict_corners_from_image(mean=None, std=None)` currently return an unstandardised
   `[0,1]` tensor and run the model on it. The networks were trained on inputs standardised
   to mean ≈0.83, std ≈0.14, so that is roughly a −2.4σ shift: the output degrades badly and
   **nothing raises**. `test_corner_pipeline_inference` calls the function without mean/std,
   so the suite passes while exercising the broken path.
2. Make the normalisation **required**, resolved from the checkpoint's own config
   (`resolve_from_checkpoint`), and raise if it cannot be determined. A silent default that
   produces a wrong-but-plausible image is worse than a missing key.
3. Update the pipeline tests to assert standardisation was applied.

### B. The two mandatory pipelines — `[REQ-29]`, `[REQ-32]`

4. **Enhancement** (spec §3.4, all four steps named): preprocess → predict → post-process
   (resize back to the original dimensions, convert to 8-bit) → visualise.
5. **Corners** (spec §5.1, all four steps named): preprocess → predict with the better
   model → **map coordinates back to the original image resolution** → overlay on the raw
   photo, colour-coded per `00-project/conventions.md` §8.
6. Both must accept a path, a PIL image or a numpy array, and handle greyscale, RGBA and
   EXIF rotation.

### C. The chained scanner — `[REQ-40]`

7. `src/pipeline/scanner.py`: raw photo → corner model → 4 corners → homography → warp →
   enhancement model → clean output. No human input anywhere in the chain.
8. **Map predicted corners back to the original photo resolution before computing the
   homography**, then warp at full resolution and resize to 512 for enhancement. Warping at
   512 and upsampling afterwards throws away detail the original photo still had.
9. Keep the warp behind a thin interface (`src/geometry/warp.py`) with a `cv2` backend.
   The `kornia` backend was only needed for the dropped Phase 09 — do not build it.

### D. Ordering safety — the failure the spec warns about

10. Spec §7 hint: "if the predicted corners are in the wrong order, the homography will flip
    or rotate the page." The enhancement output then looks catastrophic for reasons that
    have nothing to do with the enhancement network.
11. Before trusting any number, render the colour-coded overlay on **every** real photo and
    confirm the ordering by eye.
12. Add a validity check — convex, correctly ordered by cross-product sign, area above a
    floor — that **logs a warning rather than silently sorting**. Sorting hides a real
    failure and breaks on rotated pages (`conventions.md` §1).

### E. The two-way evaluation — `[REQ-41]`

13. Run the full chain on all real photos **twice**: once rectifying with the **annotated**
    corners, once with the **predicted** corners.
14. Report the OCR metric under the matched-resolution protocol (ADR-011 §5 — reuse
    `evaluate.py`'s `to_ocr_canvas`) and qualitative results for both.
15. **Interpret the difference**, do not just tabulate it: it "tells you exactly how much
    corner errors cost the enhancement stage."

### F. Robustness — `[REQ-46]`, `[REQ-49]`

16. Test all three entry points explicitly on: an odd aspect ratio, a greyscale JPEG, an
    EXIF-rotated image, a very large image, a photo where the page is small in frame, and a
    cluttered background.
17. **None may crash.** This is the acceptance criterion for the phase.
18. Rehearse the `[REQ-43]` live-modification demo: change a hyperparameter, swap the
    upsampling mode, add a degradation. The config-driven design makes each a ten-second
    demonstration — but only if you have done it once before.

---

## Gate

- [ ] Standardisation is required, not optional, in both preprocessing paths; tests assert it
- [ ] Enhancement pipeline: all four spec §3.4 steps, runs on an arbitrary unseen image
- [ ] Corner pipeline: all four spec §5.1 steps, coordinates mapped back to original resolution
- [ ] `scanner.py` runs end to end on a raw photo with no human input
- [ ] Corner ordering verified visually on **every** real photo
- [ ] Validity check present, warning rather than silently sorting
- [ ] Chain evaluated with **annotated** corners
- [ ] Chain evaluated with **predicted** corners
- [ ] The difference interpreted in prose, not just tabulated
- [ ] All six robustness cases pass without crashing
- [ ] Live-modification demo rehearsed at least once

---

## Failure modes

**Corner ordering.** The one the spec explicitly warns about, and it is easy to misdiagnose
as a model problem. Check the overlays before believing any number.

**Silently sorting predicted corners.** Makes the symptom disappear and leaves the disease.

**Warping at 512 instead of full resolution.** Throws away detail for a purely mechanical
reason and makes the end-to-end result worse than it needs to be.

**Evaluating the chain only once.** `[REQ-41]` requires both conditions; the *contrast* is
the deliverable.

**Mixing up which corners were used.** Phase 05 uses annotated corners throughout; this
phase uses both. Label every figure and metric unambiguously.

**Unstandardised input at inference.** Task A. It does not raise, and it is the failure most
likely to be discovered live, in front of the graders, on a photo you cannot control.

---

## Skills

- `05-skills/eval-integrity.md` — the two-way comparison must be clean
- `05-skills/scope-guard.md` — resist rebuilding anything here; this phase is composition

---

## Deliverables

| Artifact | Location |
|---|---|
| Enhancement pipeline | `src/pipeline/enhance.py` |
| Corner pipeline | `src/pipeline/corners.py` |
| End-to-end scanner | `src/pipeline/scanner.py` |
| Warp interface (cv2 backend) | `src/geometry/warp.py` |
| Two-way evaluation table | `outputs/reports/` |
| Annotated-vs-predicted side-by-side | `outputs/figures/p08_annotated_vs_predicted.png` |
| End-to-end results on real photos | `outputs/figures/p08_endtoend.png` |
| Robustness test notes | `state/session-log.md` |
