# Phase 00 — Foundation & Data Intake

## Objective

Get a repository that runs identically on all three machines, and find out what the data actually
is. Everything downstream is sized against numbers we do not yet have (`[OPEN-01]`), so this phase
exists to replace assumptions with measurements before any of them become load-bearing.

## Prerequisites

None. This is the entry point.

## Requirements in force

`[REQ-14]` split by scan · `[REQ-43]` config-driven, modular · ADR-001 (portability) ·
`03-spec/data-contract.md` · `03-spec/repo-layout.md`

---

## Tasks

### A. Repository skeleton
1. `git init`; copy `.agents/` in; symlink `GEMINI.md` to the root; first commit.
2. `.gitignore` per `03-spec/repo-layout.md` — **data, checkpoints and frozen sets excluded**.
3. Directory skeleton; `requirements.txt` (permitted deps only — check the "not permitted" list).
4. `README.md` stub with the entry points.

### B. Config and portability (ADR-001)
5. `src/utils/config.py`: YAML loading with `base → env → exp` layering; writes the **resolved**
   config to the run directory.
6. `configs/env/{local_cpu,mx330,colab_t4}.yaml` — device, batch size, workers, AMP.
   **AMP true only on `colab_t4`.**
7. `DATA_ROOT` / `RUNS_ROOT` from env var or a gitignored local `paths.yaml`. **No absolute paths
   in committed code.**
8. `src/utils/seeding.py`: global seeds plus `worker_init_fn` (`00-project/conventions.md` §5).
9. `notebooks/colab_train.ipynb`: mount Drive → clone → install → symlink → run. **Launcher only,
   no logic.**

### C. Data intake audit — the substance of this phase
10. Get the provided scans onto a machine and into `$DATA_ROOT/scans/`.
11. Run the full audit in `03-spec/data-contract.md` §4.
12. Download DTD into `$DATA_ROOT/backgrounds/dtd/`.
13. **Write the inventory to `state/discoveries.md`** and update every number that depended on the
    `~200 scans` assumption: split counts, frozen-set sizing, RAM cache budget.

### D. Split assignment
14. `src/data/splits.py` — 80/10/10 **by hash of filename**, not by shuffled index.
15. Emit `splits.json`; verify disjointness; check for near-duplicate scans that would leak content
    across the boundary. **Commit `splits.json`.**

### E. Benchmark (`[ASM-01]`)
16. A short identical benchmark on the MX330 and on Colab — a few hundred steps of a small conv net
    at 512×512. Record seconds/step on each. Validates or corrects the "15–20×" estimate.

### F. Kick off the human-dependent work
17. **Immediately** hand the human the capture briefs — real photos (`[OPEN-02]`,
    `01-decisions/open-questions.md`) and backgrounds (`[OPEN-03]`, ADR-004 §1). These have the
    longest latency in the project and gate Phase 01, the ADR-004 calibration, and every real-photo
    metric. Do this on **day one**, not when Phase 01 starts.

---

## Gate

- [ ] Repo clones and runs on this workstation (CPU) and on Colab, from the README alone
- [ ] `python -c "import src.utils.config"`-style smoke check passes on both
- [ ] Config layering verified: the same exp config resolves to different batch/device per env
- [ ] No absolute path anywhere in committed code (`grep -rn "/home/" src/ configs/` is empty)
- [ ] **Data audit complete**, every checklist item in `data-contract.md` §4 answered
- [ ] Inventory written to `state/discoveries.md` with actual counts and resolutions
- [ ] Assumptions that the real numbers invalidate are updated **and flagged in `STATUS.md`**
- [ ] `splits.json` generated, committed, disjointness asserted
- [ ] DTD downloaded and counted
- [ ] MX330-vs-Colab benchmark recorded; `[ASM-01]` marked validated or corrected
- [ ] Capture briefs delivered to the human; `[OPEN-02]`/`[OPEN-03]` reflected in `STATUS.md`

---

## Failure modes

**Building against assumed data.** The most likely error in this phase. If scan count, resolution or
aspect ratios differ materially from `data-contract.md` §3, several downstream decisions move.
**Audit first, then size.**

**Absolute paths.** Works on one machine, breaks on the other two, usually discovered mid-training
on Colab. The `grep` check above is worth automating.

**Scans that are not clean.** They are the *targets*. If they are skewed, shadowed or low-resolution,
they cap achievable quality no matter how good the model is. Look at 20 of them; do not trust the
filename.

**EXIF rotation.** A photo that displays upright in a viewer can load sideways in `cv2.imread`,
which ignores the orientation tag. This silently misaligns annotations later. Check now.

**Duplicate or near-duplicate scans across splits.** Breaks `[REQ-14]`'s intent even if the
filenames differ — the same page scanned twice on either side of the split is leakage.

**Deferring the human tasks.** Photos and annotation have days of latency. Deferring them to Phase 01
serialises the project unnecessarily.

---

## Skills

- `05-skills/portable-training.md` — the three-environment setup
- `05-skills/scope-guard.md` — before adding any dependency

---

## Deliverables

| Artifact | Location |
|---|---|
| Runnable repo skeleton | repo |
| Env profiles | `configs/env/` |
| Colab launcher notebook | `notebooks/` |
| Data inventory | `state/discoveries.md` |
| Split assignment | `splits.json` (committed) |
| Benchmark numbers | `state/discoveries.md` |
| Updated sizing assumptions | `state/assumptions.md` |
