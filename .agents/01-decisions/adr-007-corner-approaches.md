# ADR-007 — Both Corner Approaches, Fairly Compared

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Low (`[REQ-30]`)

## Context

`[REQ-30]` (spec §5) is unambiguous:

> "There are two natural formulations of this problem, and — this is the interesting part —
> **you will implement both and let the experiments decide which one wins.**"

The research report says the opposite. Its Phase 2 instructs "strictly abandoning the direct
regression methodology," and it presents heatmap regression as the one to adopt.

**The research report is wrong on this point, and this ADR exists mainly to nail that down.** The
report is arguing about which model to *deploy* in the final pipeline. The spec is requiring both to
be *built and measured*. Those are different claims, and an implementation agent skimming the report
could easily drop half a mandatory deliverable.

There is a second, subtler risk. Everyone — the spec's own framing, the report, the literature, and
the baseline's 8% vs 96% result — expects Approach B to win. That expectation makes it very easy to
implement Approach A carelessly, get a bad number, and call it a confirmed hypothesis. A rigged
comparison is worth no marks and is straightforwardly dishonest.

## Decision

### 1. Both approaches are built, trained, and reported. Non-negotiable.

Approach B winning is a *permitted outcome*, not a permitted shortcut.

### 2. Approach A gets a genuinely fair shot

Specific commitments:

- **Same encoder** as Approach B (`03-spec/model-specs.md` builds one configurable encoder used by
  both). Any accuracy difference then comes from the *head and the output representation*, which is
  the actual question, not from one model having a better backbone.
- **Same training budget:** same epochs, same optimizer family, same data, same frozen eval sets.
- **Same LR search effort.** If you tune Approach B's learning rate over three values, tune A's over
  three values. Record both searches.
- **Sensible design, not a straw man.** Normalised coordinates in `[0,1]` with a sigmoid output
  (ADR-009); L1 on coordinates (spec permits L1 or L2 — L1 is more robust to outliers and is the
  better-faith choice).
- **Preserve some spatial layout in the flatten.** Do *not* use global average pooling before the
  FC head. GAP discards all spatial information, which would make the task nearly impossible and
  would be sandbagging. Flatten a small spatial grid (e.g. 8×8 or 16×16 × C) so the FC layer can
  still read position. See `03-spec/model-specs.md` for the shape.

### 3. Pre-register the prediction

Spec §5.1 hint: "Think about *why* the two approaches might behave differently before running the
experiments, and **write your prediction down.** … Was your prediction right?"

**Before training either model**, write the prediction and its reasoning into
`state/discoveries.md`, dated. Then report whether it held. This is explicitly asked for and it is
free marks.

The expected reasoning, to be stated in the agent's own words and then tested: fully connected
layers destroy spatial topology, forcing a global feature vector to be mapped to precise
coordinates — a hard, shift-sensitive mapping. Heatmap regression keeps the problem spatial and
local, so a fully convolutional network preserves the translation-equivariance that localisation
naturally wants. Prior evidence: the baseline scored 8.00% (A) vs 96.00% (B) on synthetic data.

### 4. The comparison must answer all three of the spec's questions

`[REQ-31]` asks three things, and the report must answer each with evidence:

| Question | Evidence to collect |
|---|---|
| Which is **more accurate**? | Mean corner error + success rate, synthetic test *and* real photos |
| Which is **more robust to unusual viewpoints**? | Stratify the synthetic test set by perspective severity and by page scale; report error per stratum. Also inspect the hardest real photos. |
| Which was **easier to train**? | Epochs to converge, LR sensitivity, stability across seeds, whether it needed babysitting. Record this *during* training — it cannot be reconstructed afterwards. |

Plus failure-case visualisations from **both** models, not just the loser's.

### 5. The pipeline uses the winner

`[REQ-32]` says the corner inference pipeline uses "your better trained model". Whichever wins on
the evidence goes into the pipeline and into the bonus chain. If B wins as expected, A remains in
the codebase and in the report as the comparison — it is not deleted.

## Consequences

**Good.** Satisfies a mandatory requirement that the advisory document would have led you to skip.
The comparison is credible because the fairness measures are recorded in advance rather than
asserted afterwards.

**Costs.** Roughly double the corner-detection training. Mitigated by corner models being much
cheaper to train than the enhancement network (smaller output, faster convergence).

**Risk.** Approach A may fail to train at all (the baseline's 8% success and 10.41 px mean error
suggest near-failure). **That is a legitimate, reportable result** — but only after
`05-skills/training-diagnostics.md` has been run against it, so the report can distinguish
"regression is fundamentally ill-suited here" from "we had a bug". Document which conclusion the
evidence supports.

## Notes

- Approach B's heatmap design is ADR-008.
- Approach A predicts 8 numbers, ordered `[x0,y0, x1,y1, x2,y2, x3,y3]` per
  `00-project/conventions.md` §1. Never sort the outputs to enforce ordering — that hides real
  errors and breaks on rotated pages.
- A known Approach-A failure mode from the literature and the report: with no ordering constraint,
  regression models can emit corners in an inconsistent order, producing a homography that folds
  the image. The metrics will catch this if you do *not* sort. Include an example in the failure
  visualisations if it occurs.
