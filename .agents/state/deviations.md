# Deviations

Every departure from a `[REQ]`, `[CON]`, or ADR — proposed, approved, or rejected.

**The rule (`GEMINI.md` §5):** you may not silently depart from a requirement, constraint or
decision. If you believe one is wrong, infeasible or blocked:

1. **Stop implementing.**
2. Write an entry here.
3. Put it to the human with a recommendation.
4. **Wait.** Work on something else unblocked meanwhile.
5. Implement only after approval; mark `APPROVED`; if it changes a standing decision, write a **new
   ADR that supersedes the old one** — never edit the old ADR to erase its reasoning.

**Weight of the tiers:**
- **`[REQ]` / `[CON]`** — from the graded specification. **Almost never approvable.** If one looks
  impossible, it is far more likely a misreading. Check before escalating.
- **ADR** — genuinely negotiable when the evidence has changed. That is what ADRs are for.

**Why write it down even when the answer is obvious:** an undocumented deviation is indistinguishable
from a bug six weeks later, and the report must be able to state honestly what was and was not
implemented as specified.

Template: `state/templates/deviation-entry.md`.

---

## Status values

- **`PROPOSED`** — written up, awaiting the human. **Work is paused on this item.**
- **`APPROVED`** — human agreed; implement and record the outcome
- **`REJECTED`** — human declined; the original requirement or decision stands
- **`WITHDRAWN`** — the agent resolved it without needing the deviation (record why — the reasoning
  is useful)

---

## Register

| ID | Status | Departs from | Summary | Date |
|---|---|---|---|---|
| — | — | — | *no deviations recorded* | — |

---

## Anticipated deviation points

Places where a deviation is *plausible*. Listed so they are recognised as deviations when they
arise, rather than taken as ordinary engineering choices. **None of these are pre-approved.**

| Likely trigger | Departs from | Pre-analysed position |
|---|---|---|
| Colab quota makes the 512 matrix infeasible | ADR-002 | Ladder in ADR-002 §"If compute becomes the binding constraint": reduce epochs → ablate at 256 → corners at 256. **Bring measured evidence** (s/epoch, GPU util, projected total) |
| Generator cannot feed the GPU even after ADR-003's ladder | `[REQ-11]` | A pre-generated pool with fresh photometrics applied per `__getitem__` preserves the requirement's intent. **Measure first** — do not reach for this speculatively |
| A TA rules BatchNorm is "explicit regularisation" | `[CON-04]` / ADR-005 | Swap to GroupNorm or InstanceNorm. Localised, but it invalidates trained checkpoints — **ask early** (`[OPEN-09]`) |
| Heatmaps collapse; weighted MSE is insufficient | ADR-008 | Adaptive Wing Loss. **Foreground-weighted MSE is already pre-approved and needs no deviation** — only AWL does |
| Time runs short before Phase 09 | ADR-012 | **Skipping Phase 09 is not a deviation** — it is the designed conditional behaviour. Record the reason in `STATUS.md` |
| Frozen sets must be regenerated after Phase 04 starts | ADR-003 | Bump `frozen_version`, log it, re-run affected comparisons. **Never mix versions in one table** |

---

## Entries

*(Newest first. Append below this line.)*
