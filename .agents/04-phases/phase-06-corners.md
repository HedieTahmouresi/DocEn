# Phase 06 — Corner Detection: Approach A and Approach B

## Objective

Build **both** corner-detection formulations, train them fairly, and produce the empirical
comparison the spec requires. Task 2 of the two mandatory tasks.

> ⚠️ **The research report says to abandon Approach A. It is wrong.** `[REQ-30]`: "you will
> implement both and let the experiments decide which one wins." Approach B may *win*; it may not be
> *skipped*. See ADR-007.

## Prerequisites

Phase 03 gate passed. Phase 01 complete (annotated real photos for the real-photo half of the
comparison). Independent of Phases 04/05 — interleave around GPU availability.

## Requirements in force

`[REQ-30]`, `[REQ-31]`, `[REQ-32]`, `[REQ-44]`, `[REQ-45]` · `[CON-01]`, `[CON-02]`, `[CON-04]` ·
ADR-007, ADR-008 · `02-research/corner-localization.md`

---

## Tasks

### A. Pre-register the prediction — **before training anything**
1. Spec §5.1 hint: "Think about *why* the two approaches might behave differently before running the
   experiments, and **write your prediction down.**"
2. Write it into `state/discoveries.md`, **dated**, in your own words, with reasoning. Report
   afterwards whether it held. This is explicitly asked for and it is free marks.

### B. Approach A — direct regression
3. `CornerRegNet`: shared encoder → reduce to 8×8 → flatten → FC(512) → FC(256) → FC(8) → sigmoid.
4. **Do not use global average pooling before the FC head.** GAP discards all spatial information;
   Approach A would then fail for a reason unrelated to the actual question and the comparison
   would be worthless. This is ADR-007's central fairness commitment.
5. Loss: L1 on normalised coordinates. `weight_decay=0`, no dropout (`[CON-04]`).
6. **Check the `Linear(32768, 512)` initialisation.** Poor scaling there can stall training and be
   misread as "regression doesn't work" — which would corrupt the `[REQ-30]` comparison.

### C. Approach B — heatmap regression
7. `CornerHeatmapNet`: the same Encoder+Decoder as the enhancement net, 4-channel head → sigmoid.
8. Targets: σ=8 px, peak exactly 1.0, **rendered in a ±3σ window and pasted** (ADR-008).
   Border: **clip the window, never shift it.**
9. Loss: MSE first.
   **Known risk:** the Gaussian covers ~0.7% of pixels, so the loss is background-dominated.
   *Signature:* heatmaps collapse toward zero, loss drops fast then plateaus, argmax returns noise.
   **Pre-approved fallback** (log as an experiment, no escalation): foreground-weighted MSE with
   pixel weight `1 + w·target`, `w ≈ 10–50`. Adaptive Wing Loss needs escalation.
10. Extraction: `argmax` → **local soft-argmax in an 11×11 window**. Record peak activation as a
    confidence score.
11. `[REC]` σ sweep {4, 8, 12} (`[ASM-05]`) if compute allows.

### D. Fair training — ADR-007 §2
12. Same encoder, same epochs, same optimizer family, same data, same frozen sets.
13. **Same LR search effort for both.** If you try three LRs for B, try three for A. Record both
    searches.
14. **Record "easier to train" evidence during training** — epochs to converge, LR sensitivity,
    stability, restarts. `[REQ-31]` asks for this and it **cannot be reconstructed afterwards.**

### E. Metrics — `[REQ-31]`
15. `src/metrics/corners.py`: mean corner error (px @512 **and** % of diagonal), success rate at
    **1%** and **2%** of diagonal, `[REC]` quadrilateral IoU.
16. Evaluate both models on the **synthetic test set** and the **real photos** — always as a pair.
17. **Robustness:** stratify the synthetic test set by perspective severity and page scale; report
    error per stratum. Plus the hardest real photos.

### F. Inference pipeline — `[REQ-32]`
18. `src/pipeline/corners.py`: preprocess → predict with the **better** model → **map coordinates
    back to original resolution** → overlay on the raw photo. All four steps are named in spec §5.1.
19. Test on unseen images with odd aspect ratios, greyscale, and EXIF rotation.

### G. Hard-negative ablation — `[OPEN-07]`, `[REC]`
20. Only if real-photo corner accuracy disappoints: enable distractor quadrilaterals in the
    generator (ADR-004 §2) and retrain. **If accuracy is already good, skip it and report that
    decision** — that is the scope-guard call, and skipping is itself a finding.

---

## Gate

- [ ] Prediction pre-registered in `state/discoveries.md`, dated, **before training**
- [ ] Both models trained; assertions confirm no dropout, `weight_decay=0`, no pretrained weights
- [ ] **Approach A does not use GAP** before the FC head — fairness verified by reading the code
- [ ] Equal LR search effort for both, both searches recorded
- [ ] Loss curves for both (`[REQ-22]`)
- [ ] Metrics table complete: both models × {synthetic test, real photos} × {mean err px, % diag,
      success@1%, success@2%, IoU}
- [ ] **Robustness stratification** by perspective severity and scale, reported
- [ ] **"Easier to train"** evidence recorded during training, not reconstructed
- [ ] **Failure-case visualisations for both models**, not just the loser's
- [ ] Predicted-corner overlays on real photos, colour-coded (`conventions.md` §8)
- [ ] Heatmap visualisation figure (the four channels for a sample)
- [ ] Written verdict answering all three of `[REQ-31]`'s questions
- [ ] Pre-registered prediction revisited: did it hold?
- [ ] Corner pipeline runs on unseen images including the three edge cases

---

## Failure modes

**Skipping Approach A.** The most likely scope failure in the whole project, because the research
report tells you to. It is half a mandatory deliverable.

**Sandbagging Approach A.** Everyone expects B to win, so a weak A produces the expected answer and
feels like confirmation. GAP before the FC head, an unequal LR search, or a broken init would all do
it. The comparison is the deliverable — a rigged one is worth nothing.

**Heatmap collapse read as a bug.** All-zero heatmaps are the *documented* signature of the
foreground/background imbalance. Recognise it, apply the pre-approved weighted MSE, log it.

**Sorting predicted corners to fix ordering.** Hides a real failure mode and breaks on rotated
pages. `conventions.md` §1. If ordering is wrong, the metrics should show it.

**High synthetic score believed uncritically.** The baseline scored 96% synthetic and 0% real. If
Approach B exceeds ~95% synthetic success early, **check the generator's ranges before believing
it** (`02-research/baseline-failure-analysis.md`).

**Full-frame Gaussian rendering.** ~100× slower than the windowed version, and enough on its own to
starve the GPU on Colab.

**Shifting the Gaussian window at the border.** Moves the peak, corrupts the label — and only for
images where a corner is near the edge, which is exactly the hard cases.

**Reaching for architecture when real performance is poor.** Priority order in
`sim2real-playbook.md` §3: backgrounds → homography ranges → photometric ranges → σ/loss →
architecture. Architecture is almost never the answer here.

**Comparing our pixel errors to the baseline's 1.85/107.44 px.** Its resolution and threshold are
undocumented. Only the pattern transfers.

---

## Skills

- `05-skills/training-diagnostics.md` — especially the heatmap-collapse entry
- `05-skills/experiment-discipline.md` — the fairness protocol
- `05-skills/eval-integrity.md` — the comparison must be honest
- `05-skills/scope-guard.md` — before adopting Adaptive Wing or corner refinement

---

## Deliverables

| Artifact | Location |
|---|---|
| Both architectures | `model.py` |
| Corner metrics | `src/metrics/corners.py` |
| Corner pipeline | `src/pipeline/corners.py` |
| Two trained checkpoints (+ variants) | `runs/exp-*/` |
| Comparison table | `outputs/reports/` |
| Loss curves | `outputs/figures/p06_curves_*.png` |
| Failure cases, both models | `outputs/figures/p06_failures_*.png` |
| Heatmap visualisation | `outputs/figures/p06_heatmaps.png` |
| Predicted-corner overlays | `outputs/figures/p06_overlays.png` |
| Robustness stratification | `outputs/reports/` |
| Pre-registered prediction + outcome | `state/discoveries.md` |
