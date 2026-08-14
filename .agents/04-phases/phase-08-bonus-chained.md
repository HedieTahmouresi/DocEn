# Phase 08 — Bonus: The Chained End-to-End Scanner

## Objective

Compose the two inference pipelines into a single automatic scanner — raw photo in, clean scan out,
no human input — and measure exactly what corner prediction error costs the enhancement stage.

This is **Tier 1**, the actual stated bonus (ADR-012). Mostly glue code plus an evaluation pass.

## Prerequisites

Phase 07 gate passed. Both pipelines working independently. Mandatory deliverables complete — the
bonus must not displace them.

## Requirements in force

`[REQ-40]` compose the pipelines · `[REQ-41]` evaluate **twice**: annotated vs predicted corners ·
`[REQ-46]` robustness · ADR-012

---

## Tasks

### A. The chain
1. `src/pipeline/scanner.py`: raw photo → corner model → 4 corners → homography → warp → enhancement
   model → clean output.
2. Map predicted corners **back to the original photo resolution** before computing the homography,
   then warp at full resolution and resize to 512 for enhancement. Warping at 512 and upsampling
   afterwards throws away detail the original photo still had.
3. Warp behind a thin two-backend interface — `cv2` now, `kornia` later (ADR-012). Both are
   explicitly sanctioned by spec §7.
4. Use the **winning** corner model from Phase 06.

### B. Ordering safety — the known failure mode
5. Spec §7 hint: "if the predicted corners are in the wrong order, the homography will flip or
   rotate the page."
6. **Before trusting any number**, render the colour-coded overlay (`conventions.md` §8) on every
   real photo and confirm the ordering.
7. Add a validity check: quad convex, corners correctly ordered by cross-product sign, area above a
   sensible floor. **Log a warning rather than silently "fixing" it by sorting** — sorting hides a
   real failure and breaks on rotated pages.

### C. The two-way evaluation — `[REQ-41]`
8. Run the full chain on all real photos **twice**:
   - **(a)** rectifying with the **annotated** corners
   - **(b)** rectifying with the **predicted** corners
9. Report the OCR metric (matched-resolution protocol, ADR-011 §5) and qualitative results for both.
10. Table per `03-spec/evaluation-spec.md` §7.
11. **Interpret the difference**, do not just tabulate it: it "tells you exactly how much corner
    errors cost the enhancement stage."

### D. Analysis
12. Which photos degrade most under predicted corners? Correlate the CER drop with that photo's
    corner error and with the heatmap confidence from Phase 06.
13. Is there a corner-error threshold beyond which enhancement quality falls off sharply? A scatter
    of corner error vs CER answers this directly and is a strong figure.
14. Any catastrophic failures (flipped or rotated pages)? Include them — they are the most
    informative cases.

### E. Robustness — `[REQ-46]`, `[REQ-49]`
15. The TAs will run this on photos you have never seen. Test explicitly: odd aspect ratios,
    greyscale JPEGs, EXIF-rotated images, a very large image, a photo where the page is small in
    frame, a photo on a cluttered background.
16. **It must not crash.** A pipeline that throws on an unusual input loses more than one that
    returns a mediocre result.

---

## Gate

- [ ] `scanner.py` runs end-to-end on a raw photo with no human input
- [ ] Coordinates mapped to original resolution before the homography; warp at full res
- [ ] Corner ordering verified visually on **every** real photo
- [ ] Validity check present, warning rather than silently sorting
- [ ] Full chain evaluated with **annotated** corners
- [ ] Full chain evaluated with **predicted** corners
- [ ] Comparison table complete; the difference **interpreted in prose**
- [ ] Side-by-side figure: the same photo rectified both ways
- [ ] Corner-error vs CER scatter produced
- [ ] Robustness cases all tested; **no crashes**
- [ ] Any flip/rotation failures documented rather than hidden

---

## Failure modes

**Corner ordering.** The one the spec warns about. A flipped page makes the enhancement output look
catastrophic for reasons entirely unrelated to the enhancement network — and it is easy to
misdiagnose as a model problem. **Check the overlays first, always.**

**Silently sorting predicted corners.** Makes the symptom disappear while leaving the disease, and
breaks on rotated pages.

**Warping at 512 instead of full resolution.** Throws away detail the original photo still had, and
makes the bonus result worse than it needs to be for a purely mechanical reason.

**Evaluating only once.** `[REQ-41]` explicitly requires both conditions. The *contrast* is the
deliverable — a single number does not satisfy it.

**Mixing up which corners were used.** Phase 05 uses annotated corners; this phase uses both.
Label figures and metrics unambiguously or the comparison becomes unreadable.

**Crashing on an unusual input.** `[REQ-49]` means this runs on unseen photos in front of the
graders. Handle greyscale, EXIF rotation, and extreme aspect ratios.

**Starting Phase 09 from here without checking ADR-012's conditions.** The joint fine-tune is
conditional. Verify the mandatory work is complete first.

---

## Skills

- `05-skills/eval-integrity.md` — the two-way comparison must be clean
- `05-skills/scope-guard.md` — resist building the joint training loop here

---

## Deliverables

| Artifact | Location |
|---|---|
| End-to-end scanner pipeline | `src/pipeline/scanner.py` |
| Warp interface (cv2/kornia backends) | `src/geometry/warp.py` |
| Two-way evaluation table | `outputs/reports/` |
| Annotated-vs-predicted side-by-side | `outputs/figures/p08_annotated_vs_predicted.png` |
| End-to-end results on real photos | `outputs/figures/p08_endtoend.png` |
| Corner-error vs CER scatter | `outputs/figures/p08_error_vs_cer.png` |
| Robustness test notes | `state/session-log.md` |
