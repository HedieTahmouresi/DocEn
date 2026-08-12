# Template: Deviation Entry

Append to `state/deviations.md` and add a row to its register.

**Write this BEFORE implementing anything.** Then ask the human and wait
(`06-workflow/escalation-protocol.md`).

---

```markdown
## DEV-NNN — <short title>

Status:       PROPOSED
Date:         YYYY-MM-DD
Departs from: [REQ-nn] | [CON-nn] | ADR-nnn
Phase:        NN

**What the requirement/decision says:**
Quote it. Do not paraphrase — the exact wording usually matters.

**Why I believe it should change:**
Evidence, with numbers. Not intuition, not "it would be cleaner".

**What I propose instead:**
Specific and implementable.

**Risk of departing:**
Including whether it invalidates prior results, checkpoints, or comparability.

**Risk of NOT departing:**
The other half. If this is small, you probably should not be proposing it.

**Reversible?**
If we do this and it is wrong, what does it cost to undo?

**What I am doing meanwhile:**
Unblocked work. "Waiting" is not an acceptable answer.
```

---

## Worked example

```markdown
## DEV-001 — Reduce corner-detector resolution to 256

Status:       PROPOSED
Date:         2026-09-02
Departs from: ADR-002 (512x512 for all three networks)
Phase:        06

**What the decision says:**
"512x512 everywhere. Enhancement network, Approach A, Approach B, and every ablation
including the loss comparison and the dropout study."

**Why I believe it should change:**
Measured: a corner run at 512 takes 3.2 h on Colab T4 (exp-010). Phase 06 needs 4 runs
(A, B, and dropout variants of each) plus a sigma sweep = ~16 h minimum. Free-tier
quota has given ~4 h/day over the past week, and I have been throttled twice. Projected
completion for Phase 06 alone: 4+ days, which puts Phases 07-08 at risk.
At 256 a corner run is ~50 min (measured on a truncated run).

**What I propose instead:**
Corner detectors only at 256; enhancement stays at 512. Coordinates are normalised to
[0,1] (REQ-13), so mapping back to full resolution is exact. ADR-002 lists this as
fallback option 3 and calls it "the best-justified reduction on the merits" —
corner detection keys on global quadrilateral geometry, not text detail.

**Risk of departing:**
- Report must state two resolutions and justify the split. Modest cost.
- Heatmap quantisation error roughly doubles (~+1 px at 256 before soft-argmax).
- exp-010 and exp-011 would need re-running at 256 for a consistent comparison (~2 h).
- The human explicitly chose 512 partly out of concern that 256 causes problems.
  That concern is the main argument against.

**Risk of NOT departing:**
Phases 07 and 08 may not complete. Phase 07 is MANDATORY (REQ-38/39). Trading a
mandatory deliverable for uniform resolution is the wrong trade.

**Reversible?**
Yes — re-running at 512 later costs ~16 h of quota. Nothing is destroyed.

**What I am doing meanwhile:**
Continuing Phase 06 Approach A at 512 (already queued), and drafting the Phase 05
report sections, which need no GPU.
```

---

## What makes a deviation likely to be approved

- **Measured evidence**, not anticipation
- **Both risks stated** honestly, including the argument against yourself
- A **specific** alternative, not "do something else"
- Acknowledging what it invalidates
- A clear recommendation
- Unblocked work continuing meanwhile

## What makes one likely to be rejected

- "This would be better" with no measurement
- Departing from a `[REQ]` or `[CON]` — these come from the graded spec and are almost never
  approvable. If one looks impossible, re-read it first; it is far more likely a misreading
- Proposing it after already implementing it
- Ignoring the cost to prior results
- Reaching for it before exhausting the options the ADR itself lists
