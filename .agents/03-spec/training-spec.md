# Training Specification

**Requirements:** `[REQ-21]` train.py with train/val split · `[REQ-22]` loss curves ·
`[REQ-23]` loss choice · `[REQ-45]` comparisons
**Constraints:** `[CON-04]` no dropout, `weight_decay=0` in Phases 04 and 06

---

## 1. Config schema

Everything tunable lives here (`00-project/conventions.md` §9). The resolved config is written to
the run directory so every result is reproducible.

```yaml
run:
  experiment_id: exp-014
  name: enh-l1msssim
  seed: 1337
  notes: "L-C variant, alpha=0.84"

env:                          # from configs/env/{local_cpu,mx330,colab_t4}.yaml
  device: cuda
  amp: true                   # true on T4 only; MX330 has no usable tensor cores
  num_workers: 2

model:
  arch: unet                  # unet | corner_reg | corner_heatmap
  base_channels: 64
  levels: 4
  out_channels: 3
  upsample: transpose         # transpose | bilinear
  dropout: 0.0                # MUST be 0.0 in Phases 04 and 06 (CON-04)

data:
  resolution: 512
  samples_per_epoch: 4000
  batch_size: 16
  frozen_version: v1          # must match across compared runs

loss:
  type: l1_msssim             # mse | l1 | l1_msssim | l1_msssim_sobel
  alpha: 0.84                 # MS-SSIM weight (Zhao et al. 2017)
  sobel_weight: 0.1

optim:
  optimizer: adam
  lr: 1.0e-3
  weight_decay: 0.0           # MUST be 0.0 in Phases 04 and 06 (CON-04)
  scheduler: cosine           # constant | cosine | plateau
  epochs: 60
  grad_clip: 1.0

checkpoint:
  every_epoch: true           # Colab sessions die (ADR-001)
  keep_best_on: val_loss
  resume: null
```

---

## 2. Optimiser and schedule

**Adam**, `lr=1e-3`, `betas=(0.9, 0.999)`. Standard, robust, and it is what the U-Net literature
uses.

> ⚠️ **`weight_decay` must be 0.0** in Phases 04 and 06 — weight decay is explicit regularisation
> and is squarely covered by `[CON-04]`. **Do not use `AdamW`**, whose default is `0.01`. Use
> `Adam`, or set it explicitly and assert it at startup.

**Schedule:** cosine annealing to `lr/100` over the run. Simple, effective, no tuning. Constant LR
is also acceptable — say which you used. Avoid elaborate warm-restart schemes; `[CON-10]`.

**Gradient clipping** at norm 1.0. Cheap insurance, particularly for the MS-SSIM variants where a
near-degenerate flat patch can produce a large gradient.

**Batch size 16** at 512×512 on a T4 with AMP. Reduce to 8 if OOM. On the MX330 expect 2–4.
**Keep batch size identical across compared runs** — it changes BatchNorm statistics and therefore
the result, which would confound the loss ablation.

---

## 3. Mixed precision

On the T4 only (ADR-001). `torch.amp.autocast` + `GradScaler`.

> ⚠️ **Cast the MS-SSIM computation to `float32` explicitly.** Its products, divisions and
> small-variance terms are not fp16-safe and will produce NaN. The failure is confusing because
> everything else trains normally until the loss goes NaN mid-epoch.

Disable AMP on the MX330 (Pascal, no usable tensor cores — it adds overhead for no gain) and on CPU.

---

## 4. Epoch structure

`samples_per_epoch` is decoupled from the scan count (spec §3.2 explicitly invites this). 4000
samples/epoch at batch 16 = 250 steps. Prefer **more, shorter epochs** over fewer long ones: more
checkpoints, finer loss curves, and much better behaviour when a Colab session dies.

Per epoch:
1. Train over `samples_per_epoch` freshly generated samples.
2. Evaluate on the **frozen** validation set (`[REQ-21]`).
3. Log train loss, val loss, and val PSNR/SSIM (enhancement) or corner error (corner nets).
4. Checkpoint. Update `best.pt` if validation improved.

**Never evaluate on the test set during training** (`[CON-07]`).

---

## 5. Logging — `[REQ-22]`

The spec requires plotting train and validation loss against epochs, and calls the analysis
"essential". Log per epoch to a CSV/JSON in the run directory — **not only to stdout**, which a
disconnected Colab notebook loses.

Minimum columns: `epoch, train_loss, val_loss, val_psnr, val_ssim, lr, epoch_seconds`.
For corner runs: `val_corner_err_px, val_success_strict, val_success_lenient`.

Also record, for `[REQ-31]`'s "which was easier to train?" question, things that **cannot be
reconstructed afterwards**: epochs to convergence, whether the LR needed changing, instability,
divergence, restarts. Note them in the session log *while training*.

---

## 6. Loss implementations

Per ADR-006, four enhancement variants — identical in every other respect:

| ID | Loss | Config |
|---|---|---|
| L-A | MSE | `type: mse` |
| L-B | L1 | `type: l1` |
| L-C | `α·(1−MS-SSIM) + (1−α)·L1`, α=0.84 | `type: l1_msssim` |
| L-D | L-C + `λ·SobelL1`, λ=0.1 | `type: l1_msssim_sobel` |

**Sign:** MS-SSIM is a similarity in `[0,1]`, higher is better. The loss is `1 − MS-SSIM`. Getting
this backwards trains the model to destroy structure.

**Corner losses:**
- Approach A: L1 on normalised coordinates
- Approach B: MSE on heatmaps; foreground-weighted MSE is the pre-approved fallback (ADR-008)

---

## 7. Checkpointing and resume

Colab sessions die. Non-negotiable (ADR-001):

- Save every epoch: model state, optimizer state, scheduler state, epoch, RNG state, resolved config.
- Keep `last.pt` and `best.pt`.
- `--resume` restores all of the above and continues.
- Write checkpoints to **Drive**, not the Colab local disk, which vanishes.
- **Test resume before the first long run** — a resume path that only gets exercised after a crash
  is a resume path that does not work.

---

## 8. Expected behaviour and rough budgets

Numbers to sanity-check against, not targets to hit.

| | Enhancement | Corner A | Corner B |
|---|---|---|---|
| Converges by | ~40–60 epochs | ~30–50 | ~30–50 |
| Val PSNR (synthetic) | should clearly beat the `[REQ-26]` baseline | — | — |
| Corner err (synthetic) | — | expect worse | expect better |
| First-epoch smell test | loss drops within the first epoch | same | same |

**If the loss is flat after one epoch, stop and run `05-skills/training-diagnostics.md`.** Do not
let a broken run continue for 50 epochs — on a shared free-tier GPU that is the most expensive
possible mistake.

---

## 9. Sanity ladder — run these before any long training run

In order. Each takes minutes and each catches a distinct class of bug.

1. **Forward pass**: one batch through the model; check output shape, dtype, range.
2. **Overfit one batch**: train on a single batch for ~200 steps. Loss must approach zero and the
   output must visually match the target. **If it cannot overfit one batch, it will never learn the
   dataset** — the bug is in the model, loss, or data, not in the hyperparameters. This is the
   single highest-value check in the list.
3. **Loss sanity**: compute the loss between target and itself (must be ~0) and between target and
   noise (must be large).
4. **Metric sanity**: PSNR/SSIM of an image against itself (∞ / 1.0).
5. **One short epoch**: 100 steps, confirm the loss decreases and checkpointing works.
6. **Resume test**: kill it, resume, confirm the loss continues rather than jumping.

---

## 10. Reproducibility

- One `seed` in config, recorded in every run.
- Seed `torch`, `numpy`, `random`, and per-worker RNGs (`00-project/conventions.md` §5).
- Full determinism (`torch.use_deterministic_algorithms`) is **not** required and costs speed —
  on-the-fly generation is deliberately stochastic anyway. Seed for *comparability of intent*, not
  bitwise reproduction.
- What actually guarantees comparability is the **frozen** val/test sets, not the seed.
- Record the git commit in every run directory. A metric without a commit cannot be reproduced.
