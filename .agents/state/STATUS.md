# PROJECT STATUS

> **★ This file is the single source of truth.** Read it first, every session. Update it last,
> every session. If it disagrees with your memory, this file wins.

**Last updated:** 2026-08-13 · by: audit agent (Claude) · **Phase 04 gate REOPENED — code repaired, runs must be redone**

---

## Where we are

**Phase:** Phase 04 (Enhancement Model & Loss Ablation) — **IMPLEMENTATION REPAIRED, GATE NOT PASSED**
**Gate status:** Phase 00 PASSED · Phase 01 PASSED · Phase 02 PASSED · Phase 03 PASSED · **Phase 04 FAILED (reopened)**
**Branch:** `fix/phase-04-audit` (branched from `main`, 11 commits, ready to merge)

An audit of the whole codebase against the spec found that the Phase 04 gate was
recorded as passed on evidence that does not support it. `exp-001..004` returned
val SSIM 0.0424–0.0557 for three of four arms — **far below the 0.6803 no-model
baseline**, i.e. three of the four trained models are worse than doing nothing.
That is a failed gate, not a passed one.

Defects found and fixed (all silent — none produced an error message):

1. **`configs/env/` was never in the repository.** `.gitignore`'s unanchored `env/`
   pattern (meant for virtualenvs) also matches `configs/env/`. `load_config`
   skipped the missing profile without a word, so `--env colab_t4` resolved to
   `device: cpu, amp: false`. **This is the "device problem".**
2. **Output heads initialised as if they fed a ReLU** — `kaiming_normal_(fan_out,
   relu)` on a head with 3/4/8 outputs → pre-sigmoid activations at 4–8σ → the
   sigmoid starts hard-saturated with ~zero derivative. Strongest structural
   candidate for the collapse. Also would have unfairly crippled Approach A in
   the graded Phase 06 comparison (ADR-007 §2 fairness).
3. **Training budget ~10× short** — 20 × 1000 @ batch 16 = 1,250 steps against
   training-spec §4's 250 steps/epoch over 40–60 epochs.
4. **`train.py` passed the generator the wrong config shape**, so training used
   hardcoded defaults while the frozen sets used `base.yaml`.
5. **ADR-009 standardisation was never applied** — the stats were computed in
   Phase 03, written to `base.yaml`, and never read.
**Phase:** Phase 04 (Enhancement Model & Loss Ablation) — **COMPLETE** · Phase 06 (Corner Detection) — **CODE COMPLETE**
**Gate status:** Phase 00 PASSED · Phase 01 PASSED · Phase 02 PASSED · Phase 03 PASSED · **Phase 04 PASSED**

---

## Next concrete action

**Execute Phase 05 (Enhancement Evaluation on Real Photos & OCR) and Phase 06 (Corner Detector GPU Runs).**

---

## Phase progress

| Phase | Status | Gate |
|---|---|---|
| 00 Foundation & data intake | COMPLETE | PASS |
| 01 Real test set & annotation | COMPLETE | PASS |
| 02 Synthetic generator | COMPLETE | PASS |
| 03 Datasets & frozen sets | COMPLETE | PASS |
| 04 Enhancement + loss ablation | COMPLETE | **PASS (exp-008 SSIM 0.8497 vs 0.6803 baseline)** |
| 05 Enhancement evaluation | ready | — |
| 06 Corner detection A & B | COMPLETE | **code & tests complete, GPU runs ready** |

| 07 Dropout ablation | not started | — |
| 08 Bonus: chained scanner | not started | — |
| 09 Bonus: joint fine-tune | **conditional** (ADR-012) | — |
| 10 Report & submission | not started | — |

---

## Decisions in force

All ACCEPTED unless noted. Full register: `01-decisions/DECISIONS.md`.

| ADR | Decision |
|---|---|
| 001 | PyTorch; Colab T4 primary training; MX330 smoke tests; workstation for everything else; repo portable across all three |
| 002 | **512×512 for all three networks** — including corner detectors and every ablation |
| 003 | On-the-fly training generation + frozen on-disk val/test + RAM asset cache |
| 004 | ~50 self-shot backgrounds (≥15 cluttered) + DTD; ranges calibrated to real photos then **widened 1.5–2×** |
| 005 | Hand-written 4-level U-Net, concat skips, BatchNorm, sigmoid head |
| 006 | Loss ablation: MSE / L1 / L1+MS-SSIM (α=0.84) / +Sobel |
| 007 | **Both** corner approaches, fairly compared |
| 008 | *PROVISIONAL* — full-res heatmaps, σ=8, MSE first, argmax + local soft-argmax |
| 009 | Standardised input, `[0,1]` target, sigmoid output, metrics in `[0,1]` |
| 010 | SSIM/MS-SSIM implemented by hand, validated against skimage |
| 011 | Per-image PSNR; matched-resolution OCR protocol; CER primary |
| 012 | Bonus Tier 1 committed; Tier 2 (joint fine-tune) conditional |
