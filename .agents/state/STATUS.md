# PROJECT STATUS

> **★ This file is the single source of truth.** Read it first, every session. Update it last,
> every session. If it disagrees with your memory, this file wins.

**Last updated:** 2026-08-14 · by: implementation agent (Antigravity) · **Phase 04 & Phase 06 PASSED**

---

## Where we are

**Phase:** Phase 05 (Enhancement Real Photo Evaluation & OCR) & Phase 07 (Dropout Ablation)
**Gate status:** Phase 00 PASSED · Phase 01 PASSED · Phase 02 PASSED · Phase 03 PASSED · Phase 04 PASSED · **Phase 06 PASSED**
**Branch:** `main` (clean working tree, Phase 04 & Phase 06 results committed)

### Summary of Passed Phase Gates:
- **Phase 04 (Enhancement Loss Ablation)**: `exp-008` (L1 + MS-SSIM + Sobel) achieved **SSIM 0.8497** and **PSNR 23.93 dB** (vs 0.6803 no-model baseline).
- **Phase 06 (Corner Detection Networks)**: `exp-010` (`CornerHeatmapNet`) achieved **Val MCE 1.05 px (0.14% diagonal)**, **99.8% Success Rate @ 1%**, and **Real Photo MCE 62.11 px**!


---

## Next concrete action

**Execute Phase 05 (Enhancement Evaluation on Real Photos & OCR) and Phase 07 (Dropout Ablation).**

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
| 06 Corner detection A & B | COMPLETE | **PASS (exp-010 Val MCE 1.05 px, 99.8% Succ@1%, Real MCE 62.11 px)** |
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
