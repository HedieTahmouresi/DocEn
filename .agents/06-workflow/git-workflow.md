# Git Workflow

Deliberately simple. This is a solo project with one implementation agent — the workflow exists to
make history readable and to make "which commit produced this number?" answerable, not to manage
collaboration.

---

## Branches

```
main                    always working; tagged at each phase gate
phase/NN-slug           one branch per phase
```

- Branch from `main` at the start of a phase: `git checkout -b phase/02-generator`
- Work there; commit often
- **Merge to `main` only when the phase gate passes**
- Tag the merge: `git tag phase-02-complete`

`main` should always be in a state where a clean clone runs. That is what makes the Phase 10
clean-clone test cheap instead of a scramble.

**Exception — experiment branches.** For a speculative change that might be thrown away:
`exp/NNN-slug`. Merge if it works, delete if not, and **record it in
`state/experiments.md` either way** (`experiment-discipline.md` rule 5: never delete a failed run
to tidy up).

---

## Commits

Format:

```
<type>: <short summary>

<optional body — the WHY, not the what>

Phase: NN
ADR: nnn                 (if the commit implements or is governed by a decision)
Experiment: exp-NNN      (if it relates to a training run)
Req: REQ-nn, REQ-mm      (if it implements specific requirements)
```

Types: `feat` · `fix` · `docs` · `test` · `refactor` · `exp` · `data` · `chore`

Example:

```
feat: OpenCV degradation pipeline with the six specified degradations

Order fixed per REQ-34. Ranges are provisional pending the Phase 01
calibration. Inverse homography is computed by matrix inversion, not
re-derived from corner points — re-derivation misaligns by 1-2 px and
REQ-35 depends on exact alignment.

Phase: 02
ADR: 003, 004
Req: REQ-33, REQ-34, REQ-35, REQ-36
```

**The trailers are the point.** They make `git log --grep="ADR: 006"` answer "what implements this
decision?" and `git log --grep="REQ-34"` answer "where is this requirement satisfied?" — which is
exactly what the Phase 10 sweep and the presentation need.

**Write the *why* in the body.** The diff shows what changed. Six weeks later the question is always
why.

---

## What is committed

**Yes:**
- All code, `tests/`
- `configs/` — including every experiment config
- `splits.json` — small, and the ground truth for "were these runs comparable?"
- **`.agents/`** — including all of `state/`
- `metrics.json` and `config.yaml` **copied back** from each run directory
- Small figures used in the report
- `requirements.txt`, `README.md`, `.gitignore`

**No:**
- `data/` anywhere — scans, backgrounds, real photos, DTD
- `frozen/` — regenerable from the manifest
- `runs/*/checkpoints/`, `*.pt`, `*.pth`
- Large figures and intermediate outputs
- `paths.yaml` (machine-local), `__pycache__`, `.ipynb_checkpoints`

**Committing a checkpoint once makes every future clone slow forever**, and Colab clones happen
constantly. Code by git, data and weights by Drive (ADR-001).

`.gitignore` should carry `*.pt`/`*.pth` as a blanket rule rather than relying on discipline.

---

## The state files are committed, and that is the point

`.agents/state/` is version-controlled deliberately. It is the handoff mechanism: a future agent
reads `STATUS.md` at `main` and knows where the project is.

Commit state updates **with the work they describe**, not in a separate tidy-up commit. A commit
that changes code without updating `STATUS.md` leaves the two out of sync, and `STATUS.md` is only
useful if it is trustworthy.

---

## Tags

```
phase-00-complete   …   phase-10-complete
frozen-v1                 when the frozen eval sets are generated
submission                the final submitted state
```

The `frozen-vN` tag matters more than it looks: it marks the commit that defines the evaluation sets
every subsequent comparison depends on (`experiment-discipline.md` rule 3).

---

## Session hygiene

At the end of every session (`GEMINI.md` §2):

1. Commit the work.
2. Commit the state updates.
3. Push, if a remote exists.

**Never end a session with uncommitted work.** The next agent starts from `STATUS.md` and the
repository — uncommitted changes are invisible to both.

If work is genuinely incomplete, commit it on the phase branch with a `wip:` prefix and say so in
`STATUS.md`. A messy commit is recoverable; a lost working tree is not.

---

## Remote

`[REC]` A GitHub remote is worth setting up: `git clone` into Colab is far smoother than syncing
code through Drive, and it is an off-machine backup.

If the repo is private, use a fine-grained PAT or deploy key in Colab. **Never commit a token.**

---

## Useful queries

```
git log --grep="Phase: 04"          everything done in Phase 04
git log --grep="ADR: 006"           what implements the loss decision
git log --grep="REQ-34"             where a requirement is satisfied
git log --grep="Experiment: exp-014" the code state behind a specific run
git tag -l                          phase completion history
git log --oneline main              the phase-level narrative
```

---

## Recovering context from git alone

If the state files are ever lost or stale, this reconstructs most of the picture:

```
git tag -l                          which phases completed
git log --oneline -30               recent work
git log --grep="Experiment:"        every training run and its commit
git show <tag>:.agents/state/STATUS.md    project state at that phase gate
```

That last one is the real argument for committing `state/`.
