# ADR-012 — Bonus Scope: Tier 1 Committed, Tier 2 Conditional

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** High
**Decided by:** the human.

## Context

Spec §7 contains two clearly different levels of ambition, and reading it quickly makes them look
like one task.

**Tier 1 — the stated bonus** (`[REQ-40]`, `[REQ-41]`). Compose the two existing inference
pipelines: predicted corners → homography → warp → enhancement network. Evaluate the full chain on
the real photos and report OCR and qualitative results **twice** — once rectifying with annotated
corners, once with predicted corners — so the cost of corner error is isolated. Plain
`cv2.getPerspectiveTransform`/`warpPerspective` is explicitly acceptable.

**Tier 2 — flagged as an Option** (`[REQ-42]`). "🧩 **Option:** Since kornia's warp is
differentiable, **the ambitious among you** can chain corner detector → warp → enhancement network
and fine-tune the whole system end-to-end with the enhancement loss."

Tier 1 is glue code over two trained models plus an evaluation pass — perhaps a day. Tier 2 is a new
training regime: gradients flowing through a soft-argmax and a differentiable warp into a network
that was trained for a different objective, with real instability risk. The spec's own wording
("the ambitious among you") marks it as beyond the baseline expectation.

## Decision

**Tier 1 is committed** as Phase 08 — a normal phase with a normal gate.

**Tier 2 is Phase 09, conditional.** It is attempted only when **all** of these hold:

1. Phases 00–08 are complete and their gates passed.
2. Every mandatory deliverable in `00-project/deliverables-checklist.md` §§1–4 is `[x]`.
3. Time and Colab quota remain for a training run plus a failed attempt.
4. `state/STATUS.md` shows no open blocker on mandatory work.

**Entering Phase 09 with mandatory work outstanding is a scope violation.** The bonus is worth
less than the required deliverables it would displace, and this is the most likely way for an
otherwise-good project to end up incomplete.

### Design for Tier 2 from the start (cheap insurance)

Even though Tier 2 may never be built, structure the code so it *could* be, at near-zero cost:

- **Extract coordinates with a differentiable local soft-argmax** (ADR-008) rather than a bare
  argmax. Already the decision for accuracy reasons; it also happens to keep the gradient path open.
- **Keep the warp behind a thin interface** with two backends — `cv2` for Tier 1, `kornia` for
  Tier 2 — so switching is a config flag, not a rewrite.
- **Keep models loadable independently**, so the joint model is a composition of two checkpoints
  rather than a third architecture.
- **Do not** build a joint training loop, a joint config schema, or joint checkpointing until
  Phase 09 actually starts. That is speculative work on a maybe.

### If Phase 09 does run

Guidance rather than specification, since it is conditional:

- **Initialise from the trained checkpoints.** Never train the chain from scratch — the spec says
  "fine-tune".
- **Freeze the enhancement network first**, and let gradients update only the corner detector. That
  isolates the question the spec asks ("does the corner detector improve when it is trained for
  what the pipeline actually needs?"). Unfreezing both is a second, later experiment.
- **Use a much lower learning rate** than either original run — 10–100× lower is the usual starting
  point for fine-tuning.
- **Keep the corner loss as an auxiliary term**, with weight. Optimising the enhancement loss alone
  gives the corner detector a degenerate escape: predicting a *smaller* crop can raise enhancement
  metrics while being geometrically wrong. Retaining the heatmap loss anchors it.
- **Expect instability.** The warp is only piecewise-well-behaved: as predicted corners move, the
  sampling grid moves, and gradients through `grid_sample` are noisy. If the corner detector
  diverges, that is a legitimate reported finding, not a failure to hide.
- **Answer both of the spec's questions explicitly:** did the corner detector improve, and did the
  annotated-vs-predicted gap from `[REQ-41]` shrink?

## Consequences

**Good.** Guarantees the actual stated bonus gets done. Protects mandatory work from being crowded
out. The design accommodations cost essentially nothing and are justified independently.

**Costs.** If everything goes well and time remains, a small amount of Phase 09 setup will not have
been pre-built. That is the correct trade against speculative work.

**Risk.** Tier 1 has one failure mode worth naming now: **corner ordering** (spec §7 hint). Wrong
order produces a homography that flips or rotates the page, and the enhancement result then looks
catastrophically bad for reasons that have nothing to do with the enhancement network. Phase 08's
gate includes an explicit ordering check — see `00-project/conventions.md` §8.

## Alternatives considered

- **Commit to both tiers.** Highest ceiling and the most interesting result. Rejected: it is the
  flagged Option tier, and committing to it risks the mandatory deliverables.
- **Tier 1 only, no accommodation for Tier 2.** Marginally simpler. Rejected: the accommodations
  (soft-argmax, warp interface) are justified on their own merits anyway.
- **Skip the bonus entirely.** Forfeits credit and the pipeline-integration story, which is the
  natural conclusion of the whole project. Rejected.
