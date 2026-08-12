# Phase 07 — Dropout Regularisation

## Objective

Add dropout to both models, retrain, and report the impact — with a specific, explicit answer to the
question the spec actually asks: **does the synthetic-to-real gap shrink?**

This is the phase where `[CON-04]` is lifted. It is lifted **here and nowhere earlier**.

## Prerequisites

Phases 05 and 06 gates passed. The un-regularised models and their metrics **preserved** — they are
the comparison arm, not superseded work.

## Requirements in force

`[REQ-38]` insert dropout in both models and retrain · `[REQ-39]` report the impact, specifically
the gap · `[REQ-45]` comparison with qualitative analysis · `03-spec/model-specs.md` §4

---

## Tasks

### A. Placement
1. **Approach A:** between the FC layers. Spec §6: "the fully connected layers are the classic
   place for Dropout."
2. **Enhancement net:** `Dropout2d` in the **bottleneck only**. Spec §6 invites experimentation
   ("experiment with where in the architecture it helps"); bottleneck-only is the reasoned starting
   point — dropout in early high-resolution layers destroys the fine spatial features that thin text
   strokes depend on, whereas bottleneck features are compressed and semantic.
3. **Approach B:** bottleneck, optionally the deepest encoder levels. The mechanism worth stating in
   the report: dropping bottleneck activations forces the network to infer a corner from global page
   geometry rather than from one memorised local texture — which is precisely the domain-overfitting
   attack `[REQ-39]` is asking about.
4. `[REC]` If time allows, try one alternative placement per model (e.g. decoder vs bottleneck) —
   spec §6 explicitly invites it and it strengthens the write-up.

### B. Retraining
5. `dropout: 0.2` to start; `[REC]` sweep `{0.1, 0.2, 0.3}` if compute allows.
6. **Everything else identical** to the corresponding Phase 04/06 run: same seed, architecture,
   schedule, batch size, `frozen_version`. Dropout is the *only* variable.
7. `weight_decay` stays 0.0 — `[REQ-38]` asks for dropout, not for a general regularisation sweep.
   Adding weight decay here would confound the comparison.
8. Register each run in `state/experiments.md`.
9. Retrain the **winning** corner approach at minimum; both if compute allows.

### C. The analysis — the actual deliverable
10. Build the table from `03-spec/evaluation-spec.md` §6:

    | Model | Variant | Synthetic val | Real photos | **Gap** |
    |---|---|---|---|---|
    | Enhancement | no dropout | | | |
    | Enhancement | dropout | | | |
    | Corner (winner) | no dropout | | | |
    | Corner (winner) | dropout | | | |

11. **The Gap column is the point.** `[REQ-39]`: "*does the gap between synthetic validation scores
    and real-photo test scores shrink?*"
12. **Answer it in a sentence, for each model.** A table without that sentence does not satisfy the
    requirement.
13. Qualitative comparison figure: before/after outputs for both models (`[REQ-45]`).

### D. Interpretation
14. If the gap **shrinks**: dropout is genuinely counteracting overfitting to synthetic artifacts —
    the mechanism the spec is pointing at. Say so and explain why.
15. If the gap **does not shrink**: equally reportable, and it implies the gap is driven by
    *distribution mismatch* rather than by memorisation — dropout regularises against overfitting a
    distribution, not against that distribution being wrong. **This is a strong result if argued
    well**, and it points back at `02-research/sim2real-playbook.md`.
16. If synthetic scores drop and real scores stay flat: dropout is costing capacity without buying
    generalisation. Report it.

**Do not run this expecting a particular answer.** All three outcomes are publishable within this
report; only an unstated one is not.

---

## Gate

- [ ] Dropout added to the enhancement net (bottleneck) and to both corner architectures
- [ ] All models retrained with **dropout as the only changed variable**
- [ ] `weight_decay` still 0.0; `frozen_version` unchanged
- [ ] Un-regularised checkpoints and metrics **preserved**, not overwritten
- [ ] Comparison table complete, including the **Gap** column
- [ ] **`[REQ-39]` answered explicitly in prose**, for both models
- [ ] Qualitative before/after figure produced
- [ ] Every run registered in `state/experiments.md`
- [ ] Interpretation written, whichever way the result went

---

## Failure modes

**Overwriting the un-regularised models.** `[REQ-38]` requires reporting *the difference*. Losing the
comparison arm means retraining it. Keep the Phase 04/06 run directories intact.

**Changing more than one thing.** A retrain that also touches the LR, the epoch count or the batch
size confounds the comparison and the phase produces nothing usable.

**Dropout in early enhancement layers.** Destroys thin-stroke detail. Symptom: the output looks
smeared and PSNR drops sharply. That is a placement mistake, not evidence that dropout is bad.

**Regenerating frozen sets.** Instantly invalidates the comparison to Phase 04/06 numbers.

**Reporting only the table.** The requirement is the *gap* question, in words. This is the single
most likely way to complete the work and still miss the requirement.

**Treating "no improvement" as failure.** It is a legitimate and interesting result — and arguing it
well demonstrates better understanding than a lucky improvement would.

---

## Skills

- `05-skills/experiment-discipline.md` — one variable at a time
- `05-skills/eval-integrity.md` — do not re-select the winner after seeing test numbers

---

## Deliverables

| Artifact | Location |
|---|---|
| Dropout-enabled architectures (config flag) | `model.py` |
| Retrained checkpoints | `runs/exp-*/` |
| Dropout comparison table with Gap column | `outputs/reports/` |
| Before/after qualitative figure | `outputs/figures/p07_dropout.png` |
| Written answer to `[REQ-39]` | `outputs/reports/` |
| Experiment records | `state/experiments.md` |
