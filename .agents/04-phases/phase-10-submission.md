# Phase 10 — Report, Packaging & Submission

## Objective

Turn completed work into a submission that satisfies every criterion, and prepare for the
presentation — where the grade is actually decided, on **photos you have never seen** (`[REQ-49]`).

## Prerequisites

Phases 00–08 complete. Phase 09 complete or explicitly skipped with a recorded reason.

## Requirements in force

`[REQ-43]`–`[REQ-48]` (Submission Criteria) · `[REQ-22]`, `[REQ-28]`, `[REQ-39]`, `[REQ-41]` ·
`00-project/deliverables-checklist.md`

---

## Tasks

### A. Sweep the checklist
1. Walk `00-project/deliverables-checklist.md` end to end. Mark each item `[x]` **only after
   verifying the artifact exists**, not from memory of having done it.
2. Every `[REQ]` in `00-project/requirements.md`: satisfied, or with an **approved** entry in
   `state/deviations.md`.
3. Every `[CON]` in `00-project/constraints.md`: verified against the **actual final code**.
   Re-run the quick self-check at the bottom of that file — a `grep` for banned imports takes
   seconds and a violation found by a grader is much more expensive.

### B. The report
4. **Numbers** — every table from `03-spec/evaluation-spec.md`. Each must trace to a `metrics.json`.
5. **Figures** — the full list in `deliverables-checklist.md` §2.
6. **The four analyses that are easy to complete work on and still omit:**
   - `[REQ-28]` the synthetic-vs-real relationship — "that gap is the central challenge"
   - `[REQ-31]` the corner verdict: accurate / robust / easier to train, all three
   - `[REQ-39]` **did the dropout gap shrink?** — in prose, both models
   - `[REQ-41]` what corner errors cost the enhancement stage
7. **Limitations** (`[REQ-48]`), including all of:
   - curled and folded pages (out of scope — the homography assumes a planar page)
   - extreme shadows
   - the synthetic-to-real gap
   - **the resolution ceiling on OCR**, with the x-height arithmetic from
     `02-research/evaluation-and-ocr.md`
   - what could not be tried and why (perceptual losses need pretrained weights — `[CON-02]`)
8. **Negative results.** The Sobel term if it did nothing, Approach A's difficulties, an unchanged
   dropout gap, distractor backgrounds if skipped. These cost nothing and demonstrate the
   understanding the criteria reward.
9. **Methodological disclosures**: the ADR-004 generator calibration against real-photo statistics
   (and why the deliberate widening makes it sound), the OCR matched-resolution protocol, and the
   loss formulation actually implemented (Zhao et al.'s Gaussian-weighted L1 or the simplified
   form).

### C. Package the code
10. `README.md`: setup, data layout, how to run each entry point, how to reproduce each table.
11. **Clean-clone test:** clone into a fresh directory, follow only the README, and confirm training
    starts and evaluation runs. Do it on a *different* machine if possible.
12. Remove dead code and stale experiments; keep the ones the report references.
13. Confirm all three pipelines run on arbitrary unseen images: odd aspect ratio, greyscale JPEG,
    EXIF rotation, very large image, small page in frame, cluttered background. **No crashes.**

### D. Presentation preparation — where the grade is decided
14. `[REQ-49]`: the staff will run your pipeline on **new, unseen realistic photos**. Rehearse:
    hand your scanner the hardest photos you can find and watch what happens.
15. `[REQ-43]`: "be prepared to explain and modify any part of the code if asked (e.g. adjusting
    hyperparameters, changing the model architecture, adding a new degradation)." The config-driven
    design makes each of those a live demonstration — **rehearse them**.
16. Be ready to explain, without reading the code:
    - why skip connections, and what happens without them
    - why L1 over L2, and why the L2 model may still win on PSNR
    - why heatmaps beat regression (or, if they didn't, why not)
    - what each of the six degradations simulates
    - why val/test are frozen
    - why BatchNorm is not "explicit regularisation" (`[OPEN-09]`, ADR-005)
    - why the corner detector runs at 512 (ADR-002)
    - what the synthetic-to-real gap is and what you did about it
17. **Confirm `[REQ-06]`** — the RoboFlow link is uploaded and TA-accessible.

### E. Close out the environment
18. Final `state/STATUS.md`: project complete, with a short summary of outcomes.
19. Final `state/session-log.md` entry.
20. `state/discoveries.md` reflects everything learned — this is what a future agent inherits.
21. Any `[OPEN-nn]` still open: mark `CLOSED` with the resolution, or record why it stayed open.
22. Tag the final commit.

---

## Gate

- [ ] Every item in `deliverables-checklist.md` verified `[x]`
- [ ] Every `[REQ]` satisfied or covered by an approved deviation
- [ ] Every `[CON]` verified against the final code, including the import grep
- [ ] All required tables present; every number traces to a `metrics.json`
- [ ] All required figures present
- [ ] `[REQ-28]`, `[REQ-31]`, `[REQ-39]`, `[REQ-41]` each answered **in prose**
- [ ] Limitations cover all five listed items
- [ ] Negative results documented
- [ ] **Clean-clone test passes** following only the README
- [ ] All three pipelines survive the six robustness cases without crashing
- [ ] Presentation rehearsal done, including a live config change
- [ ] `[REQ-06]` RoboFlow link confirmed accessible
- [ ] State files final; final commit tagged

---

## Failure modes

**Completing the work but missing the artifact.** The single most likely failure at this stage — a
comparison run but never plotted, a question answered in your head but not in the report. The
checklist exists for this. Verify artifacts, do not recall them.

**Missing the prose answers.** `[REQ-39]` in particular is easy to satisfy with a table and fail on
the actual requirement, which is the *sentence* about the gap.

**Untested clean clone.** Absolute paths, uncommitted files and undocumented setup steps only
surface here. Test on a different machine.

**A pipeline that crashes on an unusual input.** Discovered live, in front of the graders, on a
photo you cannot control.

**Hiding uncomfortable results.** If the full-resolution raw input OCRs better than the model
output, report it as a limitation. A well-argued limitation is worth more than a concealed one, and
graders find these.

**Deleting "failed" work.** Approach A's difficulties, the Sobel null result, an unchanged dropout
gap — all are reportable and all earn marks under "demonstrating deep understanding".

---

## Skills

- `05-skills/eval-integrity.md` — final honesty pass over every reported number
- `05-skills/scope-guard.md` — resist last-minute "improvements" that invalidate the tables

---

## Deliverables

| Artifact | Location |
|---|---|
| Final report | `outputs/reports/` |
| Complete figure set | `outputs/figures/` |
| Packaged, documented, clean-clone-tested repo | repo |
| Presentation notes | `outputs/reports/presentation-notes.md` |
| Final state files | `state/` |
| Tagged final commit | git |
