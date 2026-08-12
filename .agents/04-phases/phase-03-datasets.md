# Phase 03 — Datasets, Splits and Frozen Evaluation Sets

## Objective

Wrap the generator in the Dataset/DataLoader machinery, freeze the evaluation sets so every model is
scored on identical images, and prove the pipeline can actually feed a GPU.

Small phase, but it is where **leakage** and **throughput** are decided — both of which are very
expensive to discover later.

## Prerequisites

Phase 02 gate passed. Provided scans present and audited.

## Requirements in force

`[REQ-11]`, `[REQ-12]`, `[REQ-13]`, `[REQ-14]`, `[REQ-15]`, `[REQ-16]`, `[REQ-17]`, `[REQ-18]` ·
`[CON-06]`, `[CON-07]` · ADR-003, ADR-009 · `03-spec/dataset-and-splits-spec.md`

---

## Tasks

### A. Dataset classes
1. `SyntheticTrainDataset` — on the fly (`[REQ-11]`), `samples_per_epoch` from config, `task` switch
   for `'enhance'` / `'corners'`.
2. `FrozenEvalDataset` — reads PNG + corners JSON, no randomness.
3. `RealPhotoDataset` — from Phase 01; **assert the degradation pipeline can never touch it**
   (`[CON-06]`).
4. `BaselineDataset` — degraded input vs clean target from the frozen test set, so `[REQ-26]`'s
   baseline goes through the same code path as the model metrics.

### B. Freezing — `[REQ-15]`
5. `src/data/freeze.py` writes val and test **to disk** (ADR-003), as **PNG** — never JPEG, which
   would apply a second uncontrolled compression on top of pipeline step ⑥.
6. Sizing: ~500 samples each, i.e. multiple degradations per held-out scan. Re-derive from the
   actual scan count found in Phase 00.
7. `manifest.json` per set: generator config, **git commit**, seed, counts, timestamp,
   `frozen_version`. This is the comparability contract.
8. Verify: two loads are byte-identical; a load on Colab matches a load on the workstation.

### C. Loaders and normalisation
9. DataLoader config per env profile; `persistent_workers=True` so the asset cache stays warm.
10. **`worker_init_fn` with per-worker RNG** (`00-project/conventions.md` §5). Then **test it**:
    pull two batches with `num_workers=4` and assert samples differ across worker boundaries.
11. Compute per-channel mean/std **once**, from the **training split only** (ADR-009), store in
    config. Using val or test here is leakage.

### D. Verification — `[REQ-18]`
12. Visualise (input, target) pairs side by side.
13. Overlay corner labels on composites, colour-coded (`conventions.md` §8).
14. Coordinate scaling round-trip test (`[REQ-12]`); normalised coords in `[0,1]` (`[REQ-13]`).
15. Split disjointness assertion, including near-duplicate detection.

### E. Throughput — the gate that protects GPU hours
16. Measure **end-to-end sustained GPU utilisation** on Colab with a real model forward/backward.
17. If below ~50%, apply ADR-003's optimisation ladder **before** any long run.

---

## Gate

- [ ] All four Dataset classes iterate a full epoch without error
- [ ] Shapes/dtypes/ranges match `conventions.md` §3 at every boundary
- [ ] (input, target) visualisation produced and inspected (`[REQ-18]`)
- [ ] Corner overlays on composites produced and inspected (`[REQ-18]`)
- [ ] Coordinate scaling round-trips exactly; normalised coords in `[0,1]`
- [ ] **No scan appears in more than one split**; near-duplicates checked
- [ ] Frozen val/test written as PNG; manifests complete with git commit and `frozen_version`
- [ ] Frozen sets byte-identical across two loads **and across machines**
- [ ] **Worker RNG test passes** — samples differ across workers
- [ ] Normalisation stats computed from training split only, stored in config
- [ ] `RealPhotoDataset` isolation from the degradation pipeline asserted in code
- [ ] **Sustained GPU utilisation ≥ ~50% on Colab**, measured and recorded

---

## Failure modes

**The worker RNG trap.** With `num_workers > 0`, workers fork with identical RNG state. Without
explicit per-worker seeding, every worker generates the *same* samples — the dataset silently
collapses to 1/N of its intended variety, and nothing errors. Loss decreases, training looks normal,
and the model overfits far faster than it should. **This is the single most likely bug in the
phase.** Test for it explicitly.

**Frozen sets saved as JPEG.** Adds a second, uncontrolled compression on top of step ⑥. Every
metric is then slightly wrong in a way that is invisible and unreproducible.

**Split leakage by near-duplicate.** Different filenames, same page. `[REQ-14]`'s intent is that no
page appears on both sides — filename disjointness is necessary, not sufficient.

**Normalisation statistics from the wrong split.** Computing mean/std over all data leaks
information from val/test. Small effect, but free to avoid.

**Regenerating frozen sets mid-project.** Invalidates comparability with every earlier run. If the
generator must change after freezing, bump `frozen_version`, record it in `state/experiments.md`,
and **never mix versions in one table.**

**Skipping the throughput gate.** "Just start training and see" on a starved pipeline turns a
6-hour Colab session into 6 hours of a GPU waiting on `cv2.warpPerspective`.

**Caching too aggressively on the workstation.** With `num_workers=4`, the asset cache exists 4×
unless shared. On 7 GB that is a plausible OOM. Check the arithmetic.

---

## Skills

- `05-skills/eval-integrity.md` — leakage and freezing
- `05-skills/portable-training.md` — throughput and env profiles

---

## Deliverables

| Artifact | Location |
|---|---|
| Dataset classes | `src/data/datasets.py` |
| Freezing script | `src/data/freeze.py` |
| Frozen val/test + manifests | `$DATA_ROOT/frozen/` |
| Normalisation statistics | `configs/base.yaml` |
| Verification figures | `outputs/figures/p03_*.png` |
| Dataset tests | `tests/test_datasets.py` |
| Throughput / GPU-util numbers | `state/discoveries.md` |
