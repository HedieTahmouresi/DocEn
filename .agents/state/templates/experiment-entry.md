# Template: Experiment Entry

Append to `state/experiments.md` **before launching the run**, and add a row to its index table.

The hypothesis must be written **first**. Written afterwards, every outcome looks expected — and
the pre-registration is exactly what spec §5.1 asks for on the corner comparison.

---

```markdown
## exp-NNN — <short name>

Phase:        NN
Hypothesis:   What you expect and WHY. Specific enough to be wrong.
Changed:      <param>: <old> -> <new>   (vs exp-MMM)   ← exactly ONE variable
Config:       runs/exp-NNN_<slug>/config.yaml
Commit:       <short hash>
frozen_ver:   vN
Machine:      Colab T4 | MX330 | workstation
Started:      YYYY-MM-DD
Result:       the numbers. Compare directly against the run named in "Changed".
Verdict:      CONFIRMED | REFUTED | INCONCLUSIVE | ABANDONED — and what it means
Notes:        training behaviour: epochs to converge, LR sensitivity, stability,
              restarts, anything surprising. [REQ-31] asks "which was easier to
              train?" and this is the ONLY place that evidence exists.
```

---

## Worked example

```markdown
## exp-014 — enhancement, L1 + MS-SSIM (alpha=0.84)

Phase:        04
Hypothesis:   L1+MS-SSIM produces visibly sharper text than L1 alone, at similar or
              slightly LOWER PSNR — PSNR is a monotone function of MSE, so it
              structurally favours the L2-trained model (exp-012). SSIM should rise
              clearly. Confirms if SSIM > 0.90 and text is visibly sharper in the
              zoomed comparison.
Changed:      loss.type: l1 -> l1_msssim   (vs exp-013)
Config:       runs/exp-014_enh-l1msssim/config.yaml
Commit:       a1b2c3d
frozen_ver:   v1
Machine:      Colab T4
Started:      2026-08-20
Result:       val PSNR 24.1 dB (exp-013: 24.6) · val SSIM 0.912 (exp-013: 0.887)
              No-model baseline: 18.2 dB / 0.601
Verdict:      CONFIRMED — SSIM up 0.025, PSNR down 0.5 dB exactly as predicted.
              Text visibly sharper in outputs/figures/p04_loss_comparison.png.
              This is the PSNR/SSIM disagreement to explain in the report ([REQ-45]).
Notes:        ~1.4x slower per epoch than L1 (MS-SSIM backward). Converged by epoch 46.
              LR 1e-3 worked first try, no search needed. Had to cast MS-SSIM to
              float32 under AMP — produced NaN at epoch 3 otherwise.
```

---

## Verdicts

| Verdict | Meaning |
|---|---|
| **CONFIRMED** | The hypothesis held. Say by how much. |
| **REFUTED** | It did not. **Keep the run** — this is a result, and it is worth marks. |
| **INCONCLUSIVE** | The difference is within noise. Say what would settle it. |
| **ABANDONED** | Stopped early. **Say why** — crash, quota, obvious bug, superseded. |

**Never delete a run to tidy up.** A REFUTED result documented well is worth more in the report than
a CONFIRMED one that nobody doubted.

---

## Before launching — pre-flight

```
[ ] Sanity ladder run, especially overfit-one-batch
[ ] Hypothesis written and specific enough to be wrong
[ ] Config diffed against the comparison run — exactly ONE intended difference
[ ] frozen_version matches the runs this will be compared against
[ ] Assertions pass: dropout == 0, weight_decay == 0  (Phases 04/06)
[ ] Checkpointing to Drive; --resume tested
[ ] Metrics stream to file, not only stdout
[ ] Runtime fits the session budget
```

The config diff is the highest-value item. Two configs differing in three places, one of them
unintentional, is the standard way an ablation quietly becomes meaningless.
