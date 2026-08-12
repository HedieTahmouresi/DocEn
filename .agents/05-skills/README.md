# Skills

Reusable procedures for situations that recur in this project. Each encodes the specific traps of
its domain — not general advice, but what goes wrong *here*.

**Load a skill when its trigger applies.** They are listed per phase in `04-phases/` because
relevance is phase-dependent.

| Skill | Load when | Mandatory in |
|---|---|---|
| [`scope-guard.md`](scope-guard.md) | About to add any component, dependency, loss term or technique not already in `03-spec/` | **Before any architecture change** |
| [`training-diagnostics.md`](training-diagnostics.md) | A network isn't learning, loss is NaN or flat, results look wrong | — |
| [`synthetic-data-qa.md`](synthetic-data-qa.md) | Building or changing the data generator | **Phase 02** |
| [`experiment-discipline.md`](experiment-discipline.md) | Before launching any training run | **Phases 04, 06, 07** |
| [`eval-integrity.md`](eval-integrity.md) | Before computing or reporting any number | **Phases 05, 06, 07, 08** |
| [`portable-training.md`](portable-training.md) | Moving work between machines; Colab setup | — |

---

## What each is really for

**`scope-guard`** — the literature contains a better version of every component here. This is the
procedure for not building them. It also guards against *drift*: silently replacing an agreed design
with something that seems better in the moment.

**`training-diagnostics`** — a ladder from cheapest to most expensive cause. The single most
valuable check is step 1: **overfit one batch**. If a model cannot overfit a single batch, no
hyperparameter will save it. Includes a symptom table for this project's specific failures — heatmap
collapse, sigmoid saturation, worker-RNG collapse, flipped pages.

**`synthetic-data-qa`** — the generator is the only component whose bugs are invisible in the loss
curve. Misalignment, a frozen parameter, or too-narrow ranges all produce healthy training and a
model that fails on real photos. This is how you find them before spending GPU hours.

**`experiment-discipline`** — this project's deliverables are *comparisons*. A comparison between
runs that differed in more than one way is worthless, and its worthlessness is usually invisible.

**`eval-integrity`** — how not to fool yourself. Every entry is a way to produce a number that is
computed correctly and substantively misleading.

**`portable-training`** — three machines, none of which can do the whole job. The workstation has no
GPU, the MX330 is slow, and Colab has the fewest CPU cores for a CPU-bound pipeline.

---

## Two skills worth reading before you think you need them

**`scope-guard`** — by the time you feel the urge to add attention gates, you have already built a
justification for it. Reading the procedure first is cheaper than unwinding the work.

**`synthetic-data-qa`** — Phase 02 costs no GPU time and determines the ceiling on everything after
it. It is the highest-leverage phase in the project, and the easiest to under-invest in because
nothing is visibly broken when you do.
