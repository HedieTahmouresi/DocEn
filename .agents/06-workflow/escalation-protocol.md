# Escalation Protocol

When to stop and ask the human, when not to, and how to ask well.

**The balance this is trying to strike:** an agent that asks about everything is useless, and an
agent that asks about nothing silently rebuilds the project into something no one agreed to. The
line is drawn by the authority model in `GEMINI.md` §3.

---

## Stop and ask

### 1. A requirement or constraint looks wrong, contradictory, or impossible
`[REQ]`/`[CON]` come from the graded spec. If one appears unsatisfiable, that is important
information — and almost certainly a misreading worth checking rather than routing around.

### 2. You want to change a decision (ADR)
ADRs are binding until formally revised. **Deviation protocol** below.

### 3. An assumption you were told to validate turns out false
`[ASM]` items in `state/assumptions.md` name their validation point. A false assumption may
invalidate a decision built on it.

### 4. A result is drastically off-expectation and the diagnostics are exhausted
After running `05-skills/training-diagnostics.md`. Not before.

### 5. You want to add a dependency, model component, or training technique
Run `05-skills/scope-guard.md` first. If it passes all five steps and still represents a real
addition, ask.

### 6. A phase gate fails twice for the same reason
Two failures usually means the problem is upstream of where you are looking.

### 7. The work needs something only the human can do
Taking photos, annotating, uploading data to Drive, running a long GPU job, sharing a link with TAs,
paying for Colab Pro.

### 8. Compute or quota blocks the plan
`[OPEN-08]`. **Do not silently shrink the experiment matrix** — dropping a required comparison is a
scope change and the human's decision.

### 9. You are about to do something hard to reverse
Regenerating frozen evaluation sets, force-pushing, deleting run directories, overwriting a
checkpoint the report depends on.

---

## Do not ask — just decide

These are yours. Deciding them yourself is the correct behaviour; asking wastes a turn and signals
the environment is over-constrained.

- File names, module boundaries, function decomposition, refactors
- Plot styling, figure layout, colour choices (**except** the fixed corner colour code in
  `00-project/conventions.md` §8)
- Logging format, progress bars, CLI ergonomics
- Hyperparameter values **inside** a range the spec gives
- Which numpy or OpenCV idiom to use
- Test structure and framework
- The order of tasks inside a phase
- Anything listed under "Decisions explicitly not made" in `01-decisions/DECISIONS.md`
- Anything tagged `[REC]` — adopt, adapt or reject, and log what you chose

**When in doubt about whether something is a decision or a detail:** if it would appear in the
report, or if another agent would need to know it to continue, log it. If it would change what gets
built, ask.

---

## The deviation protocol

For departing from a `[REQ]`, `[CON]`, or ADR.

1. **Stop implementing.** Do not build it and ask afterwards.
2. **Write the entry** in `state/deviations.md` (template in `state/templates/`):
   - what you would depart from, with its ID
   - why — the evidence, not the intuition
   - what you propose instead
   - the risk of departing **and** the risk of not departing
   - whether it is reversible
3. **Ask the human**, plainly, with a recommendation.
4. **Wait.** Work on something else unblocked in the meantime; note the parking in `STATUS.md`.
5. **After approval:** implement, mark the deviation `APPROVED`, and if it changes a standing
   decision, **write a new ADR that supersedes the old one** — never edit the old ADR to erase its
   reasoning.

**`[REQ]`/`[CON]` deviations are almost never approvable** — they come from the graded spec. ADR
deviations are genuinely negotiable when evidence has changed.

---

## How to ask well

A good escalation is short and decidable. Include:

1. **The decision needed**, in one sentence.
2. **Why it needs the human** — which authority tier it touches.
3. **The options**, with the real trade-off.
4. **Your recommendation**, and why.
5. **What you will do while waiting.**

### Bad

> The corner model isn't working well on real photos. Should I try a different architecture?

No evidence, no options, no recommendation, and it jumps to the least likely fix
(`scope-guard.md`: architecture is almost never the answer here).

### Good

> **Decision needed:** whether to enable distractor quadrilaterals in the generator (ADR-004 §2,
> currently `[0,0]`), which requires regenerating the frozen sets and invalidates comparability with
> exp-018 through exp-021.
>
> **Why ask:** it forces a `frozen_version` bump, so every earlier corner number would need re-running.
>
> **Evidence:** Approach B gets 94% success on synthetic test, 31% on real photos. The coverage plot
> (`outputs/figures/p02_coverage.png`) shows the real distribution sits inside the synthetic one for
> geometry and photometrics, so ranges are not the issue. Failures cluster on the 6 photos with
> other papers visible in frame — mean error 88 px there vs 12 px on clean backgrounds. That points
> at background clutter specifically.
>
> **Options:**
> - (a) Enable distractors, regenerate frozen sets, re-run the 4 corner experiments. ~3 Colab hours.
> - (b) Add more cluttered background *photos* only — no generator change, no `frozen_version` bump,
>   but I only have 15 clutter backgrounds and cannot easily get more without you.
> - (c) Accept it and report the limitation.
>
> **Recommendation:** (a). The evidence is specific, ADR-004 anticipated exactly this ablation, and
> a 63-point synthetic-to-real gap is the headline risk of the project (`[REQ-49]`).
>
> **Meanwhile:** continuing Phase 06 Approach A training, which is unaffected.

---

## Blocked ≠ stuck

While waiting:

- Take unblocked work from the current phase.
- Take work from a parallel phase — Phase 01 is human-latency-bound, Phase 02 can run on stand-in
  data, Phases 04 and 06 are independent.
- Write tests, documentation or report sections.
- **Record the parking in `STATUS.md`** so the next session knows what is waiting and why.

Do not sit idle, and do not guess an answer to proceed.

---

## After an answer

1. Record it — in `state/deviations.md` if it was a deviation, in
   `01-decisions/open-questions.md` if it resolved an `[OPEN]`.
2. If it settles something durable, **write an ADR**.
3. Update `STATUS.md`.
4. Then implement.

An answered question that was never written down will be asked again next session — which is exactly
the context loss this environment exists to prevent.
