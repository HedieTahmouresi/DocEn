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
| — | — | *no experiments yet* | — | — |

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
