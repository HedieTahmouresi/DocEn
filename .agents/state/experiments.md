# Experiment Registry

Every training run, registered **before it launches** (`05-skills/experiment-discipline.md`).

**Why before:** a hypothesis written afterwards is not a hypothesis — every outcome looks expected
in hindsight. Pre-registration is also what spec §5.1 asks for explicitly on the corner comparison
("write your prediction down").

**Rules:**
- `exp-NNN` — zero-padded, monotonic, **never reused**, not even for a re-run of the same idea
- One variable at a time versus the run being compared against
- Compare only within one `frozen_version`
- **Never delete a failed run to tidy up.** Negative results are results, and they are worth marks
- Every number in the report traces to a run directory here

Template: `state/templates/experiment-entry.md`.

---

## Index

| ID | Phase | Name | Changed vs | Verdict |
|---|---|---|---|---|
| exp-001 | Phase 04 | enh_mse | Baseline | **INVALID** — Val PSNR 20.34 dB, SSIM 0.7571 |
| exp-002 | Phase 04 | enh_l1 | exp-001 | **INVALID** — Val Loss 0.5851, SSIM 0.0557 (**below** the 0.6803 baseline) |
| exp-003 | Phase 04 | enh_l1msssim | exp-001 | **INVALID** — Val Loss 0.6360, SSIM 0.0438 (**below** baseline) |
| exp-004 | Phase 04 | enh_l1msssim_sobel | exp-003 | **INVALID** — Val Loss 0.6580, SSIM 0.0424 (**below** baseline) |
| exp-005 | Phase 04 | enh_mse | re-run of exp-001 on repaired code | **PASSED** — Val PSNR 24.2829 dB, SSIM 0.8398 (PSNR winner) |
| exp-006 | Phase 04 | enh_l1 | exp-005 | **PASSED** — Val PSNR 23.8118 dB, SSIM 0.8347 |
| exp-007 | Phase 04 | enh_l1msssim | exp-005 | **PASSED** — Val PSNR 24.0853 dB, SSIM 0.8491 |
| exp-008 | Phase 04 | enh_l1msssim_sobel | exp-007 | **PASSED** — Val PSNR 23.9318 dB, SSIM 0.8497 (**SSIM winner**) |
| exp-009 | Phase 06 | corner_approach_a | CornerRegNet coordinate regression | **INVALID — collapsed head, not a result.** Val MCE 224.74 px, Succ@1% 0.0% |
| exp-010 | Phase 06 | corner_approach_b | CornerHeatmapNet heatmap regression | **PASSED / WINNER** — Val MCE 1.05 px (0.14%), Succ@1% 99.8%, Real MCE 62.11 px |
| exp-011 | Phase 06 | corner_a_fixed | exp-009 + max-pool + BatchNorm1d head + lr 3e-4 | *pending* |
| exp-012 | Phase 07 | corner_b_control | exp-010 re-derived on the shared stream as the matched control | *pending* |
| exp-013 | Phase 07 | corner_b_dropout | exp-012 + bottleneck Dropout2d p=0.2 | *pending* |
| exp-014 | Phase 07 | enh_dropout | exp-008 + bottleneck Dropout2d p=0.2 | *pending* |



### Why exp-009 is marked INVALID (2026-08-14, audit)

It is **kept, not deleted**, and it is worth reporting — but it is a statement about a
broken training run, not about coordinate regression, and it must not be presented as the
`[REQ-30]` comparison.

The output does not depend on the input. Three independent tells:

- Per-corner errors are near-identical across two completely different image distributions
  — synthetic test (500 samples) gives 152.4 / 156.8 / 159.5 / 430.2 px, real photos (30)
  give 154.9 / 151.1 / 161.8 / 462.9 px.
- Validation and test MCE agree to two decimals: 224.74 and 224.75 px.
- 224.8 px is 0.44 in normalised coordinates — **worse than simply emitting the median
  quad for every image**, which lands near 0.15–0.25. The research report's own baseline
  reached 10.41 px with the same formulation.

Diagnosis, in order of confidence: (1) `AdaptiveAvgPool2d((8,8))` average-pools 4×4 blocks
of the 32×32 bottleneck, and post-BatchNorm ReLU features are half-normal, so the pooled
vector is dominated by its per-channel DC term — the letter of ADR-007's "no GAP" with much
of GAP's damage; (2) no LR search was run for either arm, and Adam at 1e-3 on a
16.8M-parameter `Linear(32768, 512)` is the standard way to kill a wide ReLU head early —
the lopsided BL column is what partial unit death looks like; (3) the FC head carries no
normalisation, and the BatchNorm'd trunk cannot compensate.

It went unnoticed because **no overfit-one-batch test exists for either corner
architecture** — the check that exists for `EnhancementNet` across all four losses, added
in the Phase 04 audit precisely because a green suite had hidden three dead models.

exp-011 is the repair, under a new ID per rule 1.

### Why exp-001..004 are marked INVALID (2026-08-13, audit)

They are **kept, not deleted** — a negative result is a result, and this one is worth reporting.
But they cannot appear in any results table, for four independent reasons:

- Three of the four scored **below the no-model baseline** (0.6803 SSIM). A model worse than
  doing nothing has not been trained; it has failed to train.
- The output heads were initialised with a ReLU-gain Kaiming draw, starting the sigmoid
  hard-saturated. `p04_loss_comparison.png` shows L-B/L-C/L-D as a flat teal field with a
  checkerboard lattice — models that barely left their initialisation.
- The schedule was 1,250 optimiser steps, roughly a tenth of the training-spec budget.
- `configs/env/` was absent from the repository, so `--env colab_t4` resolved to `device: cpu`.
  It is not established what hardware these runs actually used.

Re-runs take **new IDs** (exp-005..008) per rule 1: an ID is never reused, not even for a re-run
of the same idea. Fresh run directories also stop the re-runs overwriting the evidence above.


---

## Frozen-set versions

The comparability contract (ADR-003). Runs across different versions **cannot be compared**, and
must never appear in the same table.

| Version | Created | Generator commit | Val / Test counts | Reason for the bump |
|---|---|---|---|---|
| — | — | — | — | *not yet generated* |

---

## Planned experiment matrix

Provisional; adjust as evidence arrives.

### Phase 04 — enhancement loss ablation (`[REQ-45]`, ADR-006)

| Planned ID | Loss | Purpose |
|---|---|---|
| exp-001 | MSE | The spec's named straw man; the PSNR reference point |
| exp-002 | L1 | Isolates the pixel-loss change alone |
| exp-003 | L1 + MS-SSIM, α=0.84 | Expected winner |
| exp-004 | + Sobel, λ=0.1 | Does an explicit edge term add anything on top? |
| exp-005+ | α sweep {0.7, 0.84, 0.95} | `[ASM-04]` — does Zhao et al.'s natural-image α transfer to documents? |

**Pre-registered prediction (ADR-006):** exp-003 wins on SSIM and visible sharpness; exp-001 wins or
ties on **PSNR** while looking clearly worse (PSNR is a monotone function of MSE, so the L2-trained
model is directly optimising it). exp-004 expected roughly neutral — the Sobel term may sharpen
strokes but cannot distinguish a text edge from a noise edge. **If exp-004 loses, that is a result.**

### Phase 06 — corner detection (`[REQ-30]`, ADR-007)

| Planned ID | Model | Purpose |
|---|---|---|
| exp-010 | Approach A, direct regression | Mandatory arm — **not optional**, despite the research report |
| exp-011 | Approach B, heatmap, σ=8, MSE | Mandatory arm |
| exp-012+ | σ sweep {4, 8, 12} | `[ASM-05]` |
| exp-01x | Approach B, foreground-weighted MSE | **Pre-approved** if heatmaps collapse (ADR-008) |
| exp-01x | Distractor-background ablation | `[OPEN-07]` — only if real-photo accuracy disappoints |

**Prediction to be pre-registered in `discoveries.md` before training** (spec §5.1 hint requires
this, and it is free marks).

**Fairness commitments (ADR-007 §2) — verify before comparing:** same encoder, same budget,
**equal LR search effort**, no GAP before Approach A's FC head.

### Phase 07 — dropout (`[REQ-38]`, `[REQ-39]`)

| Planned ID | Model | Purpose |
|---|---|---|
| exp-020 | Enhancement + bottleneck dropout | vs the Phase 04 winner |
| exp-021 | Corner winner + dropout | vs its Phase 06 run |
| exp-022+ | Rate sweep {0.1, 0.2, 0.3} | If compute allows |

**The deliverable is the Gap column** — does the synthetic-val → real-test gap shrink? All three
possible outcomes are reportable; only an unstated one is not.

### Phase 09 — joint fine-tune (conditional, ADR-012)

| Planned ID | Purpose |
|---|---|
| exp-030 | Corner net fine-tuned through the differentiable warp with the enhancement loss |

---

## Experiment entries

*(Newest first. Append below this line.)*

### 2026-08-14 — exp-009 & exp-010: Phase 06 Corner Detection Paired Comparison

- **exp-009 (`corner_approach_a`)**: `CornerRegNet` (Approach A: Coordinate Regression). Shared 4-level U-Net encoder, spatial reduction to 8x8 via `AdaptiveAvgPool2d((8, 8))` (no GAP), FC head (`32768 -> 512 -> 256 -> 8`), L1 loss, Adam lr=1e-3, CosineAnnealingLR, 40 epochs.
- **exp-010 (`corner_approach_b`)**: `CornerHeatmapNet` (Approach B: Heatmap Regression). Shared 4-level U-Net encoder-decoder, 4-channel $512 \times 512$ Gaussians ($\sigma=8.0$ px), MSE loss, $11 \times 11$ local soft-argmax sub-pixel extraction, Adam lr=1e-3, CosineAnnealingLR, 40 epochs.
- **Fairness & Protocol (ADR-007)**: Identical shared data stream, identical initialisation (`init_sigmoid_head`), zero dropout (`dropout=0.0`, `weight_decay=0.0`), evaluated on frozen validation set and real smartphone photos every epoch.
- **Pre-Registered Prediction**: Logged in `.agents/state/discoveries.md` (`[REQ-31]`).

