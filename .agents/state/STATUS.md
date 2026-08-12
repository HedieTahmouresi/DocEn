# PROJECT STATUS

> **★ This file is the single source of truth.** Read it first, every session. Update it last,
> every session. If it disagrees with your memory, this file wins.

**Last updated:** 2026-08-12 · by: planning agent (Claude) · **Implementation has not started**

---

## Where we are

**Phase:** — (pre-Phase-00)
**Gate status:** n/a
**Branch:** n/a — repository not yet initialised

The `.agents/` environment is complete. No implementation code has been written. The next agent
starts at Phase 00.

---

## Next concrete action

**Set up the implementation repository and run Phase 00.**

1. Create the implementation repo (location is the human's choice — this `.agents/` directory was
   authored at `~/amirmoh/docscan/.agents/` and is meant to be **copied into** the real repo).
2. Copy `.agents/` in; symlink `.agents/GEMINI.md` → `<repo-root>/GEMINI.md`.
3. Place both source documents in `.agents/00-project/source/`:
   - `Document Scanning Enhancement.md` (**authoritative** — the graded spec)
   - `Document Scanner Implementation Plan.md` (**advisory only**)
4. `git init`; commit `.agents/` as the first commit.
5. Read `04-phases/phase-00-foundation.md` and work its task list.

**Do this on day one, before anything else:** deliver the two capture briefs to the human
(Phase 00 task F). They have the longest latency in the project and they gate Phase 01, the ADR-004
generator calibration, and every real-photo metric.

---

## Blockers

| ID | Blocker | Blocks | Needs |
|---|---|---|---|
| `[OPEN-01]` | Provided clean scans not on this machine | Phase 00 gate, all downstream sizing | **Human** — transfer the data |
| `[OPEN-02]` | Real photos, reference scans, RoboFlow annotation, TA link | Phase 01, ADR-004 calibration, all real-photo metrics | **Human** — capture + annotate |
| `[OPEN-03]` | ~50 background photos (≥15 cluttered hard negatives) | Phase 02 calibration; DTD alone can start it | **Human** — capture |

None of these block starting Phase 00. Sections A, B, D and E of Phase 00 proceed now.

---

## Awaiting the human

- [ ] Choose and create the implementation repo location
- [ ] Transfer the provided clean scans (`[OPEN-01]`)
- [ ] Capture 20–25 real photos + a commercial reference scan for each (`[REQ-02]`, `[REQ-03]`)
- [ ] Annotate corners in RoboFlow, **TL/TR/BR/BL order** (`[REQ-05]`)
- [ ] Upload the RoboFlow link to the Google Sheet (`[REQ-06]`)
- [ ] Capture ~50 background photos, ≥15 cluttered (ADR-004 §1)
- [ ] Hand-transcribe 5 documents for CER (ADR-011 §6) — can wait until Phase 05

---

## Phase progress

| Phase | Status | Gate |
|---|---|---|
| 00 Foundation & data intake | not started | — |
| 01 Real test set & annotation | not started | — |
| 02 Synthetic generator | not started | — |
| 03 Datasets & frozen sets | not started | — |
| 04 Enhancement + loss ablation | not started | — |
| 05 Enhancement evaluation | not started | — |
| 06 Corner detection A & B | not started | — |
| 07 Dropout ablation | not started | — |
| 08 Bonus: chained scanner | not started | — |
| 09 Bonus: joint fine-tune | **conditional** (ADR-012) | — |
| 10 Report & submission | not started | — |

---

## Decisions in force

All ACCEPTED unless noted. Full register: `01-decisions/DECISIONS.md`.

| ADR | Decision |
|---|---|
| 001 | PyTorch; Colab T4 primary training; MX330 smoke tests; workstation for everything else; repo portable across all three |
| 002 | **512×512 for all three networks** — including corner detectors and every ablation |
| 003 | On-the-fly training generation + frozen on-disk val/test + RAM asset cache |
| 004 | ~50 self-shot backgrounds (≥15 cluttered) + DTD; ranges calibrated to real photos then **widened 1.5–2×** |
| 005 | Hand-written 4-level U-Net, concat skips, BatchNorm, sigmoid head |
| 006 | Loss ablation: MSE / L1 / L1+MS-SSIM (α=0.84) / +Sobel |
| 007 | **Both** corner approaches, fairly compared — the research report is wrong to skip A |
| 008 | *PROVISIONAL* — full-res heatmaps, σ=8, MSE first, argmax + local soft-argmax |
| 009 | Standardised input, `[0,1]` target, sigmoid output, metrics in `[0,1]` |
| 010 | SSIM/MS-SSIM implemented by hand, validated against skimage |
| 011 | Per-image PSNR; matched-resolution OCR protocol; CER primary |
| 012 | Bonus Tier 1 committed; Tier 2 (joint fine-tune) conditional |

---

## Live risks

| Risk | Status | Mitigation |
|---|---|---|
| **Sim2real gap** — the project's central risk; a prior baseline went 96% synthetic → 0% real | open | `02-research/sim2real-playbook.md`; ADR-004 calibration; coverage plot gate in Phase 02 |
| **CPU-bound generation on Colab** (~2 vCPUs vs a T4's appetite) | open | ADR-003 asset cache + windowed Gaussian; throughput gates in Phases 02 and 03 |
| **Colab quota** — 512 everywhere is ~4× the cheap plan | watch `[OPEN-08]` | Escalate rather than silently dropping an ablation |
| **OCR resolution ceiling** — ~2.6 px x-height at 512 | known, by design | Matched-resolution protocol (ADR-011); disclosed under `[REQ-48]` |
| **Corner ordering errors** — silent, and they flip the page | open | Fixed colour code (`conventions.md` §8); verification gates in Phases 01, 02, 08 |
| **Approach A skipped** because the research report says to | open | `[REQ-30]`, ADR-007, and a Phase 06 gate item |

---

## Notes for the next agent

- **`GEMINI.md` is the contract.** Read it fully before acting.
- **`00-project/quick-reference.md`** has every fixed number, convention and silent-failure trap on
  one page. Use it instead of re-reading twelve ADRs for a value.
- The **course spec is authoritative**; the **research report is advisory** and contains at least
  one direct contradiction of it (it says to abandon Approach A — `[REQ-30]` mandates both).
- Nothing here has been validated against real data yet. Phase 00's audit exists to replace the
  assumptions in `03-spec/data-contract.md` §3 with measurements. **Several numbers in this
  environment — split sizes, frozen-set sizing, cache budget — assume ~200 scans. Verify that.**
- Phase 02 is the highest-leverage phase and costs no GPU time. Do not rush it to get to training.
