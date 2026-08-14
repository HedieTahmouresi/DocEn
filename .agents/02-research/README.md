# Research Notes

Why the decisions in `01-decisions/` are what they are. Read the relevant file before implementing
the corresponding part — the reasoning is here, the binding choice is in the ADR.

| File | Read before |
|---|---|
| [`baseline-failure-analysis.md`](baseline-failure-analysis.md) | **Phase 02** — writing the generator. Non-optional. |
| [`sim2real-playbook.md`](sim2real-playbook.md) | **Phase 02**, and again whenever a real-vs-synthetic gap appears. |
| [`loss-functions.md`](loss-functions.md) | Phase 04 — the enhancement loss and its ablation. |
| [`corner-localization.md`](corner-localization.md) | Phase 06 — both corner approaches. |
| [`evaluation-and-ocr.md`](evaluation-and-ocr.md) | Phase 05 — before computing any reported number. |
| [`source-index.md`](source-index.md) | When you need to check or cite a claim. |

---

## Evidence grading

Claims in these notes carry different weight. The distinction matters when a claim conflicts with
what you actually observe.

| Grade | Meaning |
|---|---|
| **Verified** | From a primary source we retrieved and read. Cited in `source-index.md`. |
| **Reported** | Stated by the research report or a secondary source, not independently checked. Believe it provisionally; do not build a decision on it alone. |
| **Derived** | Our own reasoning or arithmetic. Check the working if it matters — e.g. the x-height calculation in `evaluation-and-ocr.md`. |
| **Assumption** | Flagged `[ASM-nn]`, tracked in `state/assumptions.md`, with a named validation point. |

The most consequential **Reported** items are the baseline's numbers (96%/0.00%, 1.85 px/107.44 px)
and its parameter values (`margin_ratio=0.15`, `perspective_strength=0.08`). The underlying
notebook is not in our possession. **The pattern is what we rely on; the magnitudes are not
benchmarks.**

---

## Authority of the two source documents

| Document | Status |
|---|---|
| `Document Scanning Enhancement.md` | **Authoritative.** The graded specification. All `[REQ]`/`[CON]` derive from it. |
| `Document Scanner Implementation Plan.md` | **Advisory.** Genuinely useful analysis, but not binding, and it contains errors. |

### Errors and overreaches in the research report

Recorded so they are not absorbed as fact:

1. **It contradicts a mandatory requirement.** Its Phase 2 says to "strictly abandon the direct
   regression methodology." `[REQ-30]` requires implementing **both** approaches and comparing them.
   The report is arguing about which to *deploy*; the spec is requiring both to be *built*. See
   ADR-007.

2. **Unverified empirical claims.** The baseline's numbers and generator parameters are reported at
   second hand from a notebook we do not have.

3. **Irrelevant framing.** A passage about "the United Kingdom's financial and legal sectors" and
   their document-processing standards has no bearing on this project. Ignore it.

4. **It does not notice the OCR resolution ceiling.** The report argues 512 is needed for text
   fidelity — correct — but does not follow through to the consequence that whole-page text at 512
   is still far below what an OCR engine needs, and that the OCR comparison therefore needs a
   resolution-matched protocol. See `evaluation-and-ocr.md` and ADR-011.

5. **It recommends `pytorch-msssim`** without engaging with the "explain and modify any part of the
   code" requirement (`[REQ-43]`). ADR-010 decided to implement it instead.

### Where the report is right and useful

Credit where due — this is not a document to dismiss:

- The **baseline failure analysis** is the most valuable single contribution. Diagnosing 96%→0% as
  memorised spatial priors, and pointing at the fixed `margin_ratio`/`perspective_strength`, is the
  insight the whole data strategy is built on.
- **Hard negative mining in backgrounds** — correct and important (ADR-004).
- **512 over 256 for enhancement**, on Nyquist grounds — sound reasoning, and it drove ADR-002.
- **Rejecting coordinate priors as input to the enhancement network** — correct, and independently
  matches `[CON-08]`.
- **Rejecting diffusion / attention-gated skips / DocNLC as over-engineering** — the right call for
  this scope (`[CON-10]`).
- **The L1 + MS-SSIM composite loss** with the Zhao et al. weighting — correct recipe (ADR-006).
- **Multiple dataloader workers** to hide the CPU cost of on-the-fly generation — right, and
  developed further in ADR-003.
