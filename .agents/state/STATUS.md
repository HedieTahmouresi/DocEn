# PROJECT STATUS

> **★ This file is the single source of truth.** Read it first, every session. Update it last,
> every session. If it disagrees with your memory, this file wins.

**Last updated:** 2026-08-14 · **DEADLINE: 4 hours from this update.** Scope rebalanced — see DEV-004.

---

## Where we are

**Phase:** Phase 05 (Enhancement Evaluation) → Phase 07 (Dropout) → Phase 08 (Inference & End-to-End)
**Gate status:** 00 PASS · 01 PASS · 02 PASS · 03 PASS · 04 PASS (val only) · **06 REOPENED**
**Branch:** `main`

### Phase 06 gate is REOPENED

It was recorded PASSED on evidence that does not support it. Two independent reasons:

1. **exp-009 (Approach A) never trained.** Its per-corner errors are near-identical on the
   500-sample synthetic test set and the 30 real photos — 152/157/160/430 px against
   155/151/162/463 px — and its validation and test MCE agree to two decimals (224.74 /
   224.75). Two completely different image distributions cannot produce the same errors
   unless the output barely depends on the input. 224.8 px is 0.44 in normalised
   coordinates, worse than simply emitting the median quad. ADR-007's risk clause requires
   `05-skills/training-diagnostics.md` to be run before a failing Approach A is reported as
   a result; it was not. The Phase 06 file names this exact failure mode ("Sandbagging
   Approach A") and it is a rigged comparison, which is worth nothing.
2. **Most of the gate checklist is unmet:** no robustness stratification, no failure-case
   visualisations, no heatmap figure, no loss curves, no quad IoU, no LR-search record.
   No corner run has a `metrics.json`, a `config.yaml` or a git commit in its run directory.

**Approach B's result stands** (exp-010: val MCE 1.05 px, 99.8% Succ@1%, real 61.9 px).
Only the A-vs-B *comparison* is invalid.

---

## Next concrete action

**1. Run Phase 05.** On Kaggle, after `git pull`:
```
python evaluate.py --run runs/exp-008_enh_l1msssim_sobel --env kaggle
```
`evaluate.py` is new and has never been executed. It produces the `[REQ-26]` baseline
first, the four-row `[REQ-25]` table, the spec §3.3 reading, the `[REQ-27]` triplets and
the ADR-011 §5 matched-resolution OCR, and writes `metrics.json`. If it raises, fix it —
this is the largest block of missing mandatory marks and it gates Phase 07's Gap column.

**2. Then launch both GPUs.** Corner arms share one generator stream, which is the
bottleneck, so three arms cost roughly what two cost:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/exp/exp-014_enh_dropout.yaml --env kaggle &
CUDA_VISIBLE_DEVICES=1 python train_corners.py --env kaggle \
    --configs configs/exp/exp-011_corner_a_fixed.yaml \
              configs/exp/exp-012_corner_b_control.yaml \
              configs/exp/exp-013_corner_b_dropout.yaml &
wait
```
Every arm checkpoints and logs every epoch, so **a run stopped by the deadline is still a
valid comparison** — compare all arms at whatever matched epoch they reached.

**3. Report while the GPUs run.** Nothing in the report needs a GPU.

---

## Phase progress

| Phase | Status | Gate |
|---|---|---|
| 00 Foundation & data intake | COMPLETE | PASS |
| 01 Real test set & annotation | COMPLETE | PASS |
| 02 Synthetic generator | COMPLETE | PASS |
| 03 Datasets & frozen sets | COMPLETE | PASS |
| 04 Enhancement + loss ablation | COMPLETE (20 of 40 epochs) | PASS on validation |
| 05 Enhancement evaluation | **IN PROGRESS — `evaluate.py` written, not yet run** | — |
| 06 Corner detection A & B | **REOPENED** — B stands, A being re-run as exp-011 | FAIL |
| 07 Dropout ablation | configs written (exp-013, exp-014), not launched | — |
| 08 Inference & end-to-end pipeline | **RESCOPED — now mandatory, see DEV-004** | — |
| 09 Bonus: joint fine-tune | **DROPPED** (ADR-012 conditional behaviour) | — |
| 10 Report & submission | not started | — |

---

## Scope changes in force (DEV-004)

- **Phase 09 dropped.** ADR-012 makes it conditional; the entry conditions are not met and
  skipping it is the designed behaviour, not a deviation.
- **Phase 08 rescoped** from "bonus: chained scanner" to **"Inference & End-to-End
  Pipeline"**. The two inference pipelines are `[REQ-29]`/`[REQ-32]`/`[REQ-46]` and are
  *mandatory*; `[REQ-49]` says the teaching staff run them on unseen photos at the
  presentation. The chained scanner is glue over those two and comes nearly free, so it
  stays — it is now the last mandatory phase rather than a bonus.
- **Generator range fix deferred.** The generator does not cover the measured real
  distribution (rotation ±25° against an observed 40.4°; area-fraction floor 0.15 against
  an observed 0.121; `real_profile.yaml`'s `widened_generator_ranges` block is read by
  nothing). Fixing it forces a `frozen_version` bump and invalidates every existing epoch.
  With four hours left this is a `[REQ-48]` limitation, not a task.
- **Enhancement not resumed to 40 epochs.** Stopping at 20 turns out to be an asset: the
  existing checkpoints are the matched control arm for Phase 07 as long as the dropout run
  is also 20. Resuming would have destroyed that for ~0.01 SSIM.

---

## Known defects not being fixed before submission

Recorded so they are disclosed rather than discovered. Full detail in `discoveries.md`.

| ID | Defect | Disposition |
|---|---|---|
| F-02 | Kaggle notebooks regenerate frozen sets if absent, silently breaking `frozen_version` v1 | **Do not let this run.** Ship `data/frozen/` inside `data.zip` |
| F-05 | Generator ranges do not cover measured real rotation / area / aspect | `[REQ-48]` limitation |
| F-06 | `train_corners.py` bypasses `train.py`'s helpers (no `generator_config`, no `persistent_workers`, no `metrics.json`) | Latent; defaults currently agree with `base.yaml` |
| F-07 | Multi-GPU is serialised by `.item()` inside the per-arm loop | Worked around with `CUDA_VISIBLE_DEVICES` |
| F-08 | Both inference pipelines skip standardisation when handed no config | **Fix in Phase 08** — the TAs will run these |
| F-10 | 82 files dirty from CRLF churn; no git tags | Cosmetic; do not renormalise before the deadline |

---

## Decisions in force

All ACCEPTED unless noted. Full register: `01-decisions/DECISIONS.md`.

| ADR | Decision |
|---|---|
| 001 | PyTorch; **Kaggle now primary** (4 vCPUs, 2×T4) with `configs/env/kaggle.yaml`; Colab secondary |
| 002 | 512×512 for all three networks |
| 003 | On-the-fly training generation + frozen on-disk val/test + RAM asset cache |
| 004 | ~50 self-shot backgrounds + DTD; ranges calibrated then widened — **widening never wired in, see F-05** |
| 005 | Hand-written 4-level U-Net, concat skips, BatchNorm, sigmoid head |
| 006 | Loss ablation: MSE / L1 / L1+MS-SSIM (α=0.84) / +Sobel |
| 007 | **Both** corner approaches, fairly compared — **fairness not yet achieved, exp-011 is the repair** |
| 008 | *PROVISIONAL* — full-res heatmaps, σ=8, MSE, argmax + local soft-argmax |
| 009 | Standardised input, `[0,1]` target, sigmoid output, metrics in `[0,1]` |
| 010 | SSIM/MS-SSIM implemented by hand, validated against skimage |
| 011 | Per-image PSNR; matched-resolution OCR protocol; CER primary |
| 012 | Tier 1 absorbed into Phase 08 as mandatory; **Tier 2 dropped** |
