# Skill: Experiment Discipline

**Load before:** launching any training run.

This project's central deliverables are **comparisons** — four loss functions (`[REQ-45]`), two
corner approaches (`[REQ-30]`), dropout vs none (`[REQ-38]`), annotated vs predicted corners
(`[REQ-41]`). A comparison between runs that differed in more than one way is worthless, and the
fact that it is worthless is usually invisible.

---

## The five rules

### 1. One variable at a time

Everything else identical: seed, architecture, epochs, batch size, learning rate, schedule,
`frozen_version`, generator config.

**Batch size is not neutral** — it changes BatchNorm statistics and therefore the result. Changing
it "because the other run OOMed" silently confounds the comparison. Reduce it for *both* arms, or
reduce the model width instead.

### 2. Register the run before launching it

An entry in `state/experiments.md` **before** the run starts, containing the hypothesis and what
would confirm or refute it.

Writing the hypothesis first is what makes the result interpretable. Written afterwards, every
outcome looks like it was expected — and the pre-registration `[REQ-31]` asks for (spec §5.1: "write
your prediction down") is the same idea, made explicit by the spec.

### 3. Compare only within one `frozen_version`

Frozen val/test sets are the comparability contract (ADR-003). If they are regenerated, every
earlier number becomes incomparable.

If the generator must change after freezing: bump `frozen_version`, regenerate, log the event, and
**never mix versions in one table.**

### 4. Every number traces to a run directory

```
runs/<exp-id>_<slug>/
├── config.yaml        the exact resolved config
├── metrics.json       every number, machine-readable, with git commit
├── history.csv        per-epoch log
├── checkpoints/
└── figures/
```

No number appears in the report that does not exist in a `metrics.json`. If you cannot point at the
run that produced a table cell, that cell cannot be defended.

### 5. Negative results are results

A loss term that did nothing, an approach that would not converge, a placement that hurt. Log them.
They earn marks under "demonstrating deep understanding", and they stop the next agent — or you,
next session — repeating them.

**Never delete a failed run** to tidy up. Mark it, keep it.

---

## Before launching — the pre-flight

```
[ ] Sanity ladder run (03-spec/training-spec.md §9), especially overfit-one-batch
[ ] Experiment registered in state/experiments.md with a hypothesis
[ ] Config diffed against the baseline run — exactly one intended difference
[ ] frozen_version matches the runs this will be compared against
[ ] Assertions pass: dropout == 0 and weight_decay == 0 (Phases 04/06, [CON-04])
[ ] Checkpointing to Drive; --resume tested
[ ] Metrics stream to file, not only stdout
[ ] Estimated runtime fits the session budget
```

The config diff is the highest-value item. Two configs that differ in three places, one of which was
unintentional, is the standard way an ablation quietly becomes meaningless.

---

## During the run — record what cannot be reconstructed

`[REQ-31]` asks "**which was easier to train?**". That evidence exists only while training is
happening:

- Epochs to convergence
- Whether the learning rate needed changing, and how much searching it took
- Stability: spikes, divergences, restarts
- Whether it needed babysitting
- Anything surprising

Write it in the session log **as it happens**. Reconstructing "was it easy to train?" from a loss
curve six weeks later is guesswork, and the spec is asking a real question.

---

## Fair comparison — a project-specific hazard

Everyone expects Approach B to beat Approach A. That expectation makes it very easy to produce the
expected answer through a weak Approach A, and to read it as confirmation.

ADR-007's fairness commitments, restated as checks:

```
[ ] Same encoder for both — differences come from the head, not the backbone
[ ] Same epochs, optimizer family, data, frozen sets
[ ] Equal LR search effort — three LRs for B means three for A, both recorded
[ ] No GAP before Approach A's FC head (it would discard all spatial information)
[ ] Initialisation of the large Linear layer checked, not assumed
```

The same principle applies to the loss ablation: if L-C gets a hyperparameter sweep and L-A doesn't,
the comparison is about tuning effort, not about losses.

---

## Model selection

**Select on validation. Never on test.** (`[CON-07]`, spec §2.3: the test set is "touched once, at
the end".)

- Choose the best epoch by validation metric — that is what validation is for.
- Choose the winning loss variant by validation.
- Report the winner's test number **once**.
- If you find yourself re-running test after a tweak, stop. That is validation's job, and each
  re-run inflates the headline.

---

## Sequencing under limited GPU

Colab quota is finite (`[OPEN-08]`). Order runs by information gained per hour:

1. **Cheap sanity runs first.** A 10-minute smoke test that catches a bug saves a 6-hour run.
2. **The baseline arm before the variants.** Without L-A there is nothing to compare against.
3. **Corner runs during short sessions** — they are cheaper than enhancement runs and fit a
   fragmented session better.
4. **Long enhancement runs when a stable block of time is available**, with resume tested.
5. **Optional sweeps last** (α, σ, dropout rate). They refine; they do not unblock.

**If quota becomes a blocker, escalate — do not silently drop an ablation.** Cutting a required
comparison is a scope change and is the human's decision (`[OPEN-08]`).

---

## Naming

- `exp-NNN` — zero-padded, monotonic, **never reused**, even for a re-run
- A re-run with a fix gets a new ID and references the old one
- Run directory: `runs/exp-NNN_short-slug/`
- Branch: `phase/NN-slug` (`06-workflow/git-workflow.md`)

---

## The experiment entry

Template in `state/templates/experiment-entry.md`. Minimum:

```
## exp-014 — enhancement, L1+MS-SSIM (alpha=0.84)

Phase:        04
Hypothesis:   L1+MS-SSIM produces visibly sharper text than L1 alone, at similar or
              slightly lower PSNR (PSNR structurally favours the L2-trained model).
Changed:      loss.type: l1 -> l1_msssim   (single variable vs exp-013)
Config:       runs/exp-014_enh-l1msssim/config.yaml
Commit:       a1b2c3d
frozen_ver:   v1
Machine:      Colab T4
Result:       PSNR 24.1 dB (vs 24.6 for exp-013), SSIM 0.912 (vs 0.887)
Verdict:      CONFIRMED — SSIM up, PSNR slightly down as predicted; text visibly sharper
              in outputs/figures/p04_loss_comparison.png
Notes:        ~1.4x slower per epoch than L1. Converged by epoch 46. LR 1e-3 worked
              first try, no search needed.
```
