# Phase 04 — Enhancement Network & Loss Ablation

## Objective

Build the encoder-decoder from scratch, get it training, and run the four-way loss comparison that
`[REQ-45]` grades. Task 1 of the two mandatory tasks.

## Prerequisites

Phase 03 gate passed. Frozen val/test exist and will not change (`frozen_version` fixed —
regenerating mid-ablation invalidates every earlier run).

## Requirements in force

`[REQ-19]` encoder-decoder + skips, from scratch · `[REQ-20]` `model.py` · `[REQ-21]` `train.py` ·
`[REQ-22]` loss curves · `[REQ-23]` non-blurry loss · `[REQ-45]` loss comparison ·
`[CON-01]`, `[CON-02]`, `[CON-04]` · ADR-005, ADR-006, ADR-009, ADR-010

---

## Tasks

### A. SSIM/MS-SSIM — do this first
1. `src/losses/ssim.py`: SSIM and MS-SSIM by hand (ADR-010). Parameters pinned there.
2. **`tests/test_ssim.py` — validate against `skimage` to 1e-4** on: random noise, a real document,
   identical images (exactly 1.0), a known constant offset, and a **flat/uniform patch** (the
   degenerate case — a blank document margin is exactly this).
   > Pass `gaussian_weights=True, sigma=1.5, use_sample_covariance=False` to `skimage`, or a
   > *correct* implementation will fail the test against its uniform-7×7 default.
3. This is a gate item. An unvalidated SSIM silently corrupts every number in the results table
   **and** the loss that trains the winning model.

### B. Architecture — `[REQ-19]`, ADR-005
4. `model.py`: configurable `Encoder`/`Decoder`, used three ways (`03-spec/model-specs.md` §0).
5. `EnhancementNet`: 4 levels, `base=64`, DoubleConv blocks, concat skips, 1×1 head → sigmoid.
6. **Assert `dropout == 0.0`** (`[CON-04]`). Assert no import of a pre-built network (`[CON-01]`).
7. Kaiming init; BatchNorm default init.

### C. Losses — ADR-006
8. The four variants: `mse`, `l1`, `l1_msssim` (α=0.84), `l1_msssim_sobel` (λ=0.1).
9. **Read arXiv:1511.08861 §5 before implementing L-C.** The paper's mixed loss applies a Gaussian
   weighting to the L1 term. The simplified plain-L1 form is acceptable — but choose knowingly and
   state which in the report.
10. Sobel as a fixed-weight `conv2d`, not a learned layer.
11. **Sign check:** MS-SSIM is a similarity; the loss is `1 − MS-SSIM`.

### D. Training loop — `[REQ-21]`, `[REQ-22]`
12. `train.py` per `03-spec/training-spec.md`. **`Adam` with `weight_decay=0.0`** — not `AdamW`,
    whose default is 0.01 (`[CON-04]`). Assert it at startup.
13. Per-epoch: train → evaluate on **frozen val** → log → checkpoint. Never touch test (`[CON-07]`).
14. Log to CSV/JSON in the run directory, not only stdout — a disconnected Colab notebook loses cell
    output.
15. Checkpoint every epoch to Drive; implement and **test `--resume` before the first long run**.
16. AMP on T4 only, with **MS-SSIM cast to float32** — it is not fp16-safe and will produce NaN.

### E. Sanity ladder — before any long run
17. Run all six checks in `training-spec.md` §9. **The critical one: overfit a single batch to
    near-zero loss.** If the model cannot overfit one batch, it will never learn the dataset, and
    the bug is in the model/loss/data — not the hyperparameters.

### F. The ablation — `[REQ-45]`
18. Train all four variants: **identical seed, architecture, schedule, batch size, frozen sets.**
    One variable at a time.
19. Register each in `state/experiments.md` before launching.
20. `[REC]` α sweep for L-C over {0.7, 0.84, 0.95} (`[ASM-04]`) if compute allows.
21. **Select the winner on validation, never on test** (`[CON-07]`).

---

## Gate

- [ ] `test_ssim.py` passes against `skimage` to 1e-4 on all five cases
- [ ] Model asserts: `dropout == 0.0`, `weight_decay == 0.0`, no pre-built imports
- [ ] Overfit-one-batch reaches near-zero loss and the output visually matches the target
- [ ] Resume tested: kill mid-run, resume, loss continues rather than jumping
- [ ] All four loss variants trained to convergence on identical settings
- [ ] Train/val loss curves plotted for each (`[REQ-22]`)
- [ ] **The winner clearly beats the no-model baseline** on validation (`[REQ-26]` — compute the
      baseline now if not already done)
- [ ] Every run registered in `state/experiments.md` with config, commit and verdict
- [ ] Loss-comparison figure: same input through all four models, **zoomed on text**
- [ ] Frozen sets unchanged throughout — `frozen_version` identical across all four runs

---

## Failure modes

**Unvalidated SSIM.** Corrupts the metric *and* the loss simultaneously, so the error is
self-consistent and invisible. The validation test is not optional.

**`AdamW` default weight decay.** Silently violates `[CON-04]`. Assert it.

**MS-SSIM NaN under AMP.** Everything trains normally, then the loss goes NaN mid-epoch. Cast to
float32.

**MS-SSIM sign inverted.** Trains the model to maximally destroy structure. Obvious in the images,
and it has cost people days.

**Skipping the overfit-one-batch check.** The cheapest possible bug detector, routinely skipped. A
50-epoch run that was never going to work is the most expensive mistake available on a shared
free-tier GPU.

**Comparing runs that are not comparable.** Different batch size changes BatchNorm statistics;
different `frozen_version` changes the evaluation set. Either one silently confounds the ablation.

**Selecting the winner on test.** Violates `[CON-07]` and inflates the headline number.

**PSNR/SSIM disagreement read as a bug.** Expect the MSE-trained model to win PSNR while looking
worse — PSNR is a monotone function of MSE, so L-A is directly optimising it. **This is a finding to
explain in the report**, not an anomaly to fix.

**Sigmoid saturation.** Near-white regions may come out persistently slightly grey because sigmoid
approaches 1 slowly. If the output has a dull cast, that is the cause (ADR-009 lists the options).
Do not change it mid-ablation.

---

## Skills

- `05-skills/training-diagnostics.md` — **load before debugging anything**
- `05-skills/experiment-discipline.md` — **load before launching any run**
- `05-skills/scope-guard.md` — the temptation here is attention gates and deeper nets
- `05-skills/portable-training.md` — Colab sessions and resume

---

## Deliverables

| Artifact | Location |
|---|---|
| Architectures | `model.py` |
| SSIM/MS-SSIM + validation test | `src/losses/ssim.py`, `tests/test_ssim.py` |
| Loss variants | `src/losses/composite.py` |
| Training entry point | `train.py` |
| Four trained checkpoints | `runs/exp-*/checkpoints/` |
| Loss curves (per run) | `outputs/figures/p04_curves_*.png` |
| Loss-comparison figure | `outputs/figures/p04_loss_comparison.png` |
| Experiment records | `state/experiments.md` |
