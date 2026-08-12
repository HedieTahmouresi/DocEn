# Phase 09 — Bonus: Differentiable Joint Fine-Tuning

> ## ⚠️ CONDITIONAL PHASE — check the entry conditions before starting
>
> This is the spec's flagged **🧩 Option** tier ("the ambitious among you"), not the stated bonus.
> ADR-012 makes it conditional. **Entering with mandatory work outstanding is a scope violation** —
> and it is the most likely way for an otherwise-good project to end up incomplete.

## Entry conditions — all four must hold

- [ ] Phases 00–08 complete, all gates passed
- [ ] Every item in `00-project/deliverables-checklist.md` §§1–4 is `[x]`
- [ ] Time and Colab quota remain for a training run **plus a failed attempt**
- [ ] `state/STATUS.md` shows no open blocker on mandatory work

If any fails: **stop, record why in `STATUS.md`, and go to Phase 10.** Skipping this phase is a
completely acceptable outcome and costs only bonus credit.

## Objective

Chain corner detector → differentiable warp → enhancement network, and fine-tune with the
enhancement loss, to answer the spec's question: **does the corner detector improve when it is
trained for what the pipeline actually needs?**

## Requirements in force

`[REQ-42]` (bonus, Option tier) · ADR-012

---

## Tasks

### A. The differentiable chain
1. Swap the warp backend to `kornia` (`get_perspective_transform` + `warp_perspective`) — the
   interface from Phase 08 makes this a config flag. `torch.nn.functional.grid_sample` with a
   manually constructed grid is the named alternative.
2. Coordinate extraction via the **local soft-argmax** from ADR-008 — already differentiable with
   respect to heatmap values.
3. **Document the straight-through arrangement.** The `argmax` that *selects* the 11×11 window is
   not differentiable; treat the window position as fixed per forward pass. This is standard and
   defensible, but it must be **stated** rather than glossed over.
4. Verify gradients actually flow: check `corner_net.parameters()` receive non-zero gradients from
   the enhancement loss alone. **Do this before any training** — a chain that silently detaches
   trains nothing and looks like "no improvement".

### B. Fine-tuning
5. **Initialise from the trained checkpoints.** Never from scratch — the spec says "fine-tune".
6. **Freeze the enhancement network first.** Gradients update only the corner detector. This
   isolates the spec's actual question. Unfreezing both is a second, later experiment.
7. Learning rate **10–100× lower** than the original corner run.
8. **Keep the heatmap loss as an auxiliary term**, with weight. Without it there is a degenerate
   escape: predicting a *smaller* crop can raise enhancement metrics while being geometrically
   wrong. The corner loss anchors it.
9. Expect instability. Gradients through `grid_sample` are noisy — as predicted corners move, the
   sampling grid moves. Checkpoint frequently.

### C. Evaluation — answer both of the spec's questions
10. **Did the corner detector improve?** Re-run the Phase 06 corner metrics on the fine-tuned model
    (synthetic test and real photos).
11. **Did the gap shrink?** Re-run Phase 08's `[REQ-41]` annotated-vs-predicted comparison. Spec §7:
    "Does the gap you measured above shrink?"
12. Compare against the Phase 08 numbers directly — same frozen sets, same protocol.

### D. Interpretation
13. If corner error rises but enhancement quality improves: the detector has learned to predict
    crops the enhancer handles well rather than geometrically-correct crops. **That is a genuinely
    interesting finding** and exactly what the spec is probing.
14. If it diverges: report it. Instability through the warp is a legitimate, explicable result.
15. If nothing changes: report that too, with the likely reason (the corner detector may already be
    accurate enough that the enhancement loss carries no additional signal).

---

## Gate

- [ ] Entry conditions verified and recorded **before** starting
- [ ] Gradient flow verified: corner-net parameters receive non-zero gradients from the enhancement
      loss
- [ ] Straight-through arrangement documented
- [ ] Fine-tuning runs from pretrained checkpoints, enhancement net frozen, low LR
- [ ] Auxiliary corner loss retained
- [ ] Corner metrics re-measured (synthetic + real)
- [ ] `[REQ-41]` comparison re-run
- [ ] **Both spec questions answered explicitly**: did the detector improve, did the gap shrink
- [ ] Result recorded whichever way it went, including divergence
- [ ] Experiment registered in `state/experiments.md`

---

## Failure modes

**Starting with mandatory work incomplete.** The main risk of this phase, and it is a scope
violation, not a judgment call.

**Silently detached gradients.** The chain runs, the loss decreases (the enhancement net is doing
the work), and the corner detector never updates. Looks exactly like "no improvement". **Verify
gradient flow before training.**

**Training from scratch.** The spec says fine-tune. From scratch, the joint system has no reason to
converge to anything sensible.

**Dropping the auxiliary corner loss.** Opens the degenerate escape where the detector shrinks the
crop to flatter the enhancement metric.

**Learning rate too high.** Gradients through the warp are noisy; a normal LR diverges quickly.

**Unfreezing both networks immediately.** Confounds the answer to "does the *corner detector*
improve?".

**Treating divergence as failure.** It is a reportable finding about the difficulty of joint
optimisation through a geometric warp.

---

## Skills

- `05-skills/scope-guard.md` — **verify the entry conditions honestly**
- `05-skills/training-diagnostics.md` — gradient-flow debugging
- `05-skills/experiment-discipline.md`

---

## Deliverables

| Artifact | Location |
|---|---|
| Joint training script | `train_joint.py` or a `train.py` mode |
| Kornia warp backend | `src/geometry/warp.py` |
| Fine-tuned checkpoint | `runs/exp-*/` |
| Before/after corner metrics | `outputs/reports/` |
| Re-run `[REQ-41]` comparison | `outputs/reports/` |
| Written answers to both spec questions | `outputs/reports/` |
