# Session Protocol

The rituals that make the project survivable across context windows, sessions and agents.

**The test this protocol has to pass:** a different agent, with no memory of this conversation,
reads `state/STATUS.md` and can continue the work correctly. If that would fail, the protocol was
not followed.

---

## Start of session

**In order. Do not skip to the work.**

1. **`state/STATUS.md`** — current phase, gate status, **the next concrete action**.
2. **The phase file** it points to, in `04-phases/`. Read it, do not recall it.
3. **`state/session-log.md`** — the top 2 entries. What was just tried, what happened, what was
   left unfinished.
4. **`state/deviations.md`** — anything pending approval that affects your work?
5. **`01-decisions/open-questions.md`** — does an open item **block** your next action? If so, stop
   and ask the human (`escalation-protocol.md`).
6. **`git log --oneline -10`** and `git status` — does the repository match what `STATUS.md` says?
   If not, that discrepancy is your first problem.
7. **Load the skills** the phase file names.

Only then start work.

> **Never begin implementation from memory of a previous session.** Details drift; `STATUS.md` and
> the phase file are the truth. If they disagree with your memory, they win — and the disagreement
> itself is worth noting.

---

## During the session

**Log as you go, not at the end.** Some evidence cannot be reconstructed:

- Epochs to convergence, LR sensitivity, stability, restarts — `[REQ-31]` explicitly asks "which was
  easier to train?", and that is only observable while it is happening.
- Why you chose one option over another where the environment left it open.
- Anything surprising, even if you moved past it.

**Register experiments before launching them** (`05-skills/experiment-discipline.md`). The
hypothesis written afterwards is not a hypothesis.

**Commit at natural boundaries**, not once at the end. A session that dies mid-way should leave
recoverable work.

**If you get blocked:** note it in `STATUS.md`, pick up unblocked work in the same phase, and record
the parking. Blocked is not stuck.

---

## End of session — non-negotiable

Do this **before** you run out of context, not as the last thing you attempt with what remains.
If context is getting tight, stop implementing and write the state.

1. **Append to `state/session-log.md`** — template in `state/templates/session-entry.md`.
2. **Update `state/STATUS.md`:**
   - current phase and gate status
   - **the next concrete action** — specific enough to start on without re-deriving anything
   - blockers, and what would unblock them
   - anything awaiting the human
3. **`state/experiments.md`** — any run launched, finished or abandoned.
4. **`state/discoveries.md`** — anything learned that changes the plan or a number.
5. **`state/deviations.md`** — any departure from a `[REQ]`/`[CON]`/ADR, with its approval status.
6. **`state/assumptions.md`** — any `[ASM]` validated or invalidated.
7. **Commit and push** (`06-workflow/git-workflow.md`).

> A session that produced code but no state update is an **incomplete session**. The code is not the
> deliverable of a session; the code *plus a resumable state* is.

---

## Writing `STATUS.md` well

`STATUS.md` is the single most important file in `state/`. It is read first, every time.

**The "next action" field is the one that matters.** Compare:

| Bad | Good |
|---|---|
| "Continue Phase 02" | "Implement the illumination-gradient step (§8 of `synthetic-generator-spec.md`). Steps ①–③ are done and tested in `src/data/generator.py`; the shadow compositing is next. `tests/test_generator.py` currently covers alignment only." |
| "Fix the corner model" | "Approach B's heatmaps are collapsing to zero — this matches the imbalance signature in ADR-008. Apply the pre-approved foreground-weighted MSE (`w=20`) and re-run exp-021. Baseline to beat: exp-020, 41 px mean error." |
| "Waiting on data" | "Blocked on `[OPEN-01]` (provided scans). Unblocked work available: Phase 02 tasks D and E, which use stand-in images. Human has been asked; see session-log 2026-08-14." |

Write it for someone who knows nothing about what you just did.

---

## Handoff to another agent

The four files that rebuild context, in order:

1. `state/STATUS.md` — where we are
2. `state/session-log.md` — how we got here
3. `state/experiments.md` — what has been tried and what the results were
4. `state/deviations.md` — where we knowingly departed from the plan

Plus `git log` and the phase tags. **A new agent should not need to read the code to know what to do
next.**

---

## Context-window discipline

Long sessions run out of context mid-task. To make that survivable:

- **Write state early and often**, not once at the end.
- **Prefer completing a task and logging it** over starting three and logging none.
- When context gets tight: **stop implementing, write the state, commit.** An unfinished task with
  good notes is recoverable. A finished task with no notes is not — and neither is a half-finished
  one with none.
- Do not try to hold the phase file in memory. Re-read it.

---

## Session-log entry — what makes one useful

Not a diary. It answers: *could someone else continue from this?*

```
## 2026-08-14 — Phase 02: degradation steps 1-3

**Did:** implemented perspective warp with shape-then-place corner sampling,
resolution loss, and photometric adjustment. Added the degeneracy rejection loop
(convexity + min interior angle 20 deg).

**Result:** round-trip alignment PSNR = 34.2 dB with photometrics off, above the
30 dB gate. Corner overlays verified on 20 samples including extreme geometry.

**Learned:** the rejection loop fires on ~3% of samples at perspective_strength
up to 0.35 — logged, and low enough not to bias the distribution. Above 0.45 it
climbs past 15%, which would reshape the distribution, so 0.35 stays the ceiling.

**Decided:** used INTER_AREA for downscale and randomised the upscale
interpolation. Not specified anywhere; recorded here. [REC-level call.]

**Next:** illumination gradient and shadow compositing (spec §8). Nothing blocked.

**Commits:** a1b2c3d, e4f5g6h
```
