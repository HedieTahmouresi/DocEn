# Phase Plan

The project as a sequence of verifiable units of work. Each phase has an objective, tasks, a
**gate**, and a list of failure modes seen in similar work.

---

## The phase model

- **A phase is complete when its gate passes**, not when the tasks feel done. The gate is a list of
  checks with expected outcomes — run them and record the results.
- **Gates are recorded**, not remembered. On passing, write the gate results into
  `state/session-log.md`, update `state/STATUS.md`, and tag the commit
  (`06-workflow/git-workflow.md`).
- **A failing gate is information, not an obstacle.** Fix and re-run. If the *same* gate fails twice
  for the same reason, stop and escalate (`GEMINI.md` §6) — two failures usually means the problem
  is upstream of where you are looking.
- **Do not start a downstream phase to "unblock yourself" while a gate is failing.** The one
  exception is genuine parallelism (below), and it must be recorded in `STATUS.md`.

## Dependency graph

```
  Phase 00  Foundation & data intake
      │
      ├──────────────────────────────┐
      ▼                              ▼
  Phase 01  Real test set        Phase 02  Synthetic generator
  (HUMAN-GATED)                       │      (can start on stand-in data)
      │                               │
      └──────────► calibration ───────┤
                                      ▼
                              Phase 03  Datasets, splits, frozen sets
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            Phase 04  Enhancement                Phase 06  Corner detection
            + loss ablation                      Approach A & B
                    │                                   │
                    ▼                                   │
            Phase 05  Enhancement eval                  │
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                              Phase 07  Dropout ablation
                                      ▼
                              Phase 08  Bonus: chained scanner
                                      ▼
                              Phase 09  Bonus: joint fine-tune  (CONDITIONAL — ADR-012)
                                      ▼
                              Phase 10  Report & submission
```

**Genuine parallelism:**
- Phase 01 is mostly *human* work (photos, reference scans, RoboFlow annotation). Start it
  **immediately** — it gates the ADR-004 calibration and every real-photo metric, and it is the
  longest-latency item in the project.
- Phase 02 can begin against stand-in document images while the provided scans are in transit
  (`03-spec/data-contract.md` §6).
- Phases 04 and 06 are independent once Phase 03 passes. Interleave them around GPU availability —
  corner runs are cheaper and make good use of a short Colab session.

---

## Phases

| # | Phase | Gate summary | Depends on |
|---|---|---|---|
| [00](phase-00-foundation.md) | Foundation & data intake | Repo runs on all 3 machines; data audit complete | — |
| [01](phase-01-real-testset.md) | Real test set & annotation | 10–15+ photos, references, verified corner order, TA link sent | human |
| [02](phase-02-generator.md) | Synthetic generator | Round-trip >30 dB, ordering exact, stranger test, throughput measured | 00 |
| [03](phase-03-datasets.md) | Datasets, splits, frozen sets | No leakage, frozen sets stable, GPU util ≥50% | 02 |
| [04](phase-04-enhancement.md) | Enhancement net + loss ablation | Beats no-model baseline; 4 variants trained; SSIM validated | 03 |
| [05](phase-05-enhancement-eval.md) | Enhancement evaluation | Full table + real triplets + OCR protocol + pipeline | 04, 01 |
| [06](phase-06-corners.md) | Corner detection A & B | Both trained fairly; comparison answers all 3 questions | 03, 01 |
| [07](phase-07-dropout.md) | Dropout regularisation | Both models retrained; **gap question answered explicitly** | 05, 06 |
| [08](phase-08-bonus-chained.md) | Bonus: chained scanner | End-to-end works; evaluated with annotated *and* predicted corners | 07 |
| [09](phase-09-bonus-joint.md) | Bonus: joint fine-tune | **CONDITIONAL** — only if 00–08 are clean (ADR-012) | 08 |
| [10](phase-10-submission.md) | Report & submission | Every checklist item `[x]`; clean-clone run works | all |

---

## Reading a phase file

```
Objective        one paragraph: what this phase is for
Prerequisites    what must be true before starting
Requirements     the [REQ]/[CON]/ADR items in force here
Tasks            ordered work items
Gate             checks with expected outcomes — this defines "done"
Failure modes    things that go wrong here, and what they look like
Skills           which .agents/05-skills/ files to load
Deliverables     artifacts that persist beyond the phase
```

---

## Working inside a phase

- **Read the phase file at the start of every session that touches it.** Do not work from memory of
  it — `STATUS.md` and the phase file are the truth.
- **Load the named skills.** They are listed per phase because they are relevant *there*, and they
  encode the specific traps of that phase.
- **Log as you go.** Some gate evidence — "how many epochs to converge", "did the LR need changing",
  "was it unstable" — **cannot be reconstructed after the fact**, and `[REQ-31]` explicitly asks for
  it. Write it down while it is happening.
- **The task list is a plan, not a contract.** If a task turns out unnecessary or a better route
  appears, take it and note the change. Gates are the fixed part; tasks are the suggested route to
  them.

## Time and compute shape

Rough proportions, not a schedule:

| Phase | Effort | GPU |
|---|---|---|
| 00 | small | none |
| 01 | small (mostly human latency) | none |
| **02** | **large — the highest-leverage phase** | none |
| 03 | medium | brief |
| 04 | medium code, **large GPU** (4 runs) | heavy |
| 05 | medium | light |
| 06 | medium code, **large GPU** (2+ runs) | heavy |
| 07 | small code, medium GPU | medium |
| 08 | small | light |
| 09 | conditional | medium |
| 10 | medium | none |

Phase 02 deserves disproportionate care: it costs no GPU and determines the ceiling on everything
after it (`02-research/baseline-failure-analysis.md`).
