# Skill: Training Diagnostics

**Load when:** a network isn't learning, loss is NaN or flat, or results look wrong.

**The rule:** work down the ladder in order. Each step is cheap and eliminates a whole class of
cause. Do not skip to hyperparameter tuning — it is almost never the answer, and it is the most
expensive place to look.

---

## The ladder

### 0. Before anything: is the data right?

More "model bugs" in image-to-image work are data bugs than model bugs.

- **Look at a batch.** Actually render it — input and target side by side, un-standardised. Not the
  shapes, the images.
- Are input and target **aligned**? For this project, run the round-trip check
  (`03-spec/synthetic-generator-spec.md` §3) — misalignment by even a few pixels makes the task
  partly impossible and `[REQ-35]` warns about exactly this.
- Are the ranges right? Input standardised, target in `[0,1]` (ADR-009).
- Is the target accidentally degraded? Photometrics must never touch it (`[REQ-35]`).
- BGR vs RGB — a blue-tinted render means a missed conversion.
- Corners: render the colour-coded overlay (`00-project/conventions.md` §8).

### 1. Overfit a single batch — the highest-value check in this document

Train on **one batch**, repeatedly, for ~200 steps, with everything else unchanged.

- **Loss must approach zero**, and the output must visually match the target.
- **If it cannot overfit one batch, it will never learn the dataset.** The bug is in the model, the
  loss, or the data — not in the learning rate, not in the schedule, not in capacity.

This takes two minutes and eliminates most of the search space. Skipping it is how a 50-epoch run
that was never going to work gets launched.

### 2. Loss and metric sanity

- `loss(target, target)` ≈ 0.
- `loss(target, noise)` is large.
- `PSNR(x, x)` = ∞ (guarded), `SSIM(x, x)` = 1.0.
- **MS-SSIM sign:** it is a *similarity*; the loss is `1 − MS-SSIM`. Inverted, you train the model
  to destroy structure.
- Magnitudes: L1 on `[0,1]` gives ~0.01–0.1; `1 − MS-SSIM` gives ~0.0–0.5. If a term is orders of
  magnitude larger, it dominates and the others are decoration.

### 3. Gradient flow

- Print gradient norms per layer after one backward pass.
- **All zeros** → something is detached, frozen, or the loss doesn't depend on those parameters.
- **All NaN** → see §NaN below.
- **Vanishing in early layers** → check initialisation; Kaiming for conv+ReLU
  (`03-spec/model-specs.md` §5).
- **Exploding** → gradient clipping at norm 1.0 (`03-spec/training-spec.md` §2).

### 4. Learning rate

Only now. `1e-3` with Adam is the default here.

- Loss decreases then plateaus high → try `3e-4`.
- Loss oscillates or spikes → too high; try `3e-4` or `1e-4`.
- Loss barely moves → too low, **or** something above is wrong. Suspect above first.

### 5. Capacity

Last. **Only if training-split metrics themselves are poor** — that is underfitting. If training is
good and validation is poor, that is overfitting and more capacity makes it worse.

---

## Symptom table — project-specific

| Symptom | Most likely cause | Check / fix |
|---|---|---|
| **Enhancement output is uniform grey** | Model collapsed to predicting the mean | Overfit-one-batch. If that also greys, check the loss sign and the target range. |
| **Output looks like the input, unchanged** | Learning an identity map; loss too weak or LR too low | Confirm target ≠ input; check the no-model baseline — if the model ≈ baseline, it is doing nothing |
| **Text is blurry but lighting is fixed** | Working as designed for L2 | Expected with MSE — this is exactly why ADR-006 ablates L1 and MS-SSIM |
| **Output has a persistent dull/grey cast in white areas** | Sigmoid saturates slowly near 1.0 | ADR-009: scale the target to `[0.02, 0.98]`, or use a linear head. Do **not** change mid-ablation |
| **Checkerboard artifacts on text** | `ConvTranspose2d` | Switch to `Upsample(bilinear) → Conv3×3` (ADR-005) |
| **Heatmaps are all near-zero; loss drops fast then plateaus; argmax is noise** | **Foreground/background imbalance** — the Gaussian covers ~0.7% of pixels | **Documented and expected** (ADR-008). Apply the pre-approved foreground-weighted MSE, `1 + w·target`, `w ≈ 10–50` |
| **Heatmap peaks are in the right place but corners are off** | Extraction bug, or coordinate scaling | Check argmax→coordinate conversion; check `(x,y)` vs `(row,col)` (`conventions.md` §2) |
| **Corner predictions are systematically offset** | Aspect-ratio policy mismatch between train and inference | `synthetic-generator-spec.md` §4 — one policy everywhere |
| **Corners good on synthetic, terrible on real** | **The known failure** — generator too narrow | `02-research/baseline-failure-analysis.md`; run the coverage plot before touching the model |
| **Approach A won't converge at all** | Possibly real; possibly the `Linear(32768, 512)` init | Check init scaling first (`model-specs.md` §5). A genuine failure is reportable — a bug is not |
| **Rectified page is flipped or rotated** | **Corner ordering** | `conventions.md` §1. Spec §7 warns about this. Render the overlay |
| **Val loss ≪ train loss** | Frozen val is easier than the training distribution, or a leak | Compare parameter distributions; check split disjointness |
| **Val curve is noisy epoch to epoch** | Val set not actually frozen | `[REQ-15]` — verify byte-identical loads |
| **Model overfits far faster than expected** | **Worker RNG collapse** — all workers generating identical samples | `conventions.md` §5. Test: pull two batches at `num_workers=4`, assert they differ |
| **GPU utilisation < 50%** | CPU-bound generator | ADR-003's optimisation ladder |
| **Loss goes NaN mid-epoch under AMP** | MS-SSIM in fp16 | Cast the MS-SSIM computation to float32 |

---

## NaN

In order:
1. **Under AMP?** Cast MS-SSIM to float32. This is the most likely cause in this project.
2. **SSIM denominator.** C1/C2 exist to stop division by ~0 on flat patches — and a blank document
   margin is exactly a flat patch. Do not simplify them away.
3. **log or sqrt of ≤0.** PSNR's log needs an epsilon-clamped MSE.
4. **Exploding gradients.** Clip at norm 1.0.
5. **A bad sample.** Add an assertion in the Dataset that no returned tensor contains NaN/Inf — a
   degenerate homography can produce one.

Bisect by disabling AMP first. If NaN disappears, it is a precision issue, not a maths issue.

---

## "It trains but the results are bad"

Split the diagnosis first — the fix is completely different in each case:

| Training metric | Val metric | Diagnosis | Action |
|---|---|---|---|
| Poor | Poor | **Underfitting** or a bug | Ladder steps 0–3; then capacity |
| Good | Poor | **Overfitting** | More data variety; this is what Phase 07's dropout targets |
| Good | Good, but **real photos** poor | **Domain gap** | `02-research/sim2real-playbook.md` — the generator, not the model |
| Poor | Good | **Evaluation bug** | Frozen set, normalisation, or metric direction |

That last row is worth knowing: it is nearly always a bug, not a miracle.

---

## Before escalating

Escalate when a gate fails twice for the same reason, or when a result is drastically
off-expectation and the ladder is exhausted (`GEMINI.md` §6). Bring:

- What you expected, and why.
- What you observed, with the actual numbers and output.
- Which ladder steps you ran and what each showed.
- Your best hypothesis and what would test it.

"It doesn't work" is not escalation. "Overfit-one-batch reaches 0.003 so the model and loss are
fine, gradients flow, but validation loss plateaus at 0.08 while training reaches 0.01 — I think the
frozen val set is drawn from a different distribution, and here is the parameter histogram" is.
