# Template: Architecture Decision Record

New ADRs are written when: a genuine choice is settled that future work must not silently
re-litigate, or an approved deviation supersedes an existing decision.

File: `01-decisions/adr-NNN-short-slug.md`. Add a row to `01-decisions/DECISIONS.md`.

**An ADR is not a requirement.** Requirements come from the graded spec and live in
`00-project/requirements.md`. An ADR is *our* decision within the freedom the spec allows — which
is why it is movable with evidence, and a `[REQ]` is not.

---

```markdown
# ADR-NNN — <Title>

**Status:** ACCEPTED | PROVISIONAL | SUPERSEDED by ADR-MMM
**Date:** YYYY-MM-DD · **Reversibility:** Low | Medium | High
**Decided by:** <if the human made the call, say so — it matters when reweighing it later>

## Context

What decision is needed and why it is open. What the spec says, and what it leaves to us. Any
measurements or evidence that bear on it.

## Decision

What we are doing. Specific enough to implement from.

If the decision has multiple parts, number them — they get cited individually
("ADR-004 §3" is a real reference used elsewhere).

## Consequences

**Good.** What this buys.

**Costs.** What it costs, honestly. A decision with no downside was not a decision.

**Risks.** What could go wrong, and how it would be detected.

## Alternatives considered

Each with why it was rejected. **This is the most valuable section six weeks later** — it is what
stops the same alternative being re-proposed, and it is what a viva question is likely to probe.

## Validation

Any `[ASM-nn]` this rests on, where it gets checked, and what happens if it fails.
```

---

## Guidance

**PROVISIONAL** is for a decision that must be made now but rests on an unvalidated assumption.
Name the validation point and register the `[ASM]` in `state/assumptions.md`. ADR-008 is the worked
example: σ=8 is a defensible starting point that needs a sweep.

**Superseding, not editing.** When a decision changes, write a new ADR and mark the old one
`SUPERSEDED by ADR-MMM`. **Never edit the old one to erase its reasoning** — the trail of what was
believed and why is the point, and it is what makes it possible to tell a considered reversal from
drift.

**Reversibility should drive how hard you think.** Low-reversibility decisions (framework,
conventions) deserve real deliberation. High-reversibility ones (a loss weight) deserve a decision
and a measurement, not an essay.

**Record who decided.** When the human overrode a recommendation — as in ADR-002, where the more
expensive option was chosen deliberately — note it. If the decision is ever revisited, whether it
was a considered human call or an agent default changes how it should be weighed.

**Be honest in "Costs".** An ADR that lists no downsides is not recording a decision, it is
advocating. The costs section is what makes a future re-evaluation possible.
