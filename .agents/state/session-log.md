# Session Log

Append-only work journal. **Newest entry at the top.** Template:
`state/templates/session-entry.md`.

**Purpose:** a future agent reads the top 2–3 entries and knows what was just tried, what happened,
and what was left unfinished. Write for someone who knows nothing about your session.

**Rule:** every session that touches the project appends an entry. A session that produced code but
no entry is an incomplete session (`GEMINI.md` §2).

---

## 2026-08-12 — Environment design (planning agent, Claude)

**Did:** Read both source documents in full. Researched the open technical questions (loss
functions, heatmap conventions, document-localisation literature, OCR evaluation, MS-SSIM
constraints, texture datasets). Brought four decision sets to the human. Built the complete
`.agents/` environment.

**Decisions taken with the human** (full reasoning in `01-decisions/`):
- Training on **Colab T4**, with the MX330 laptop for smoke tests and the workstation (no GPU) for
  everything else — ADR-001
- **512×512 everywhere**, including the corner detectors and every ablation. The human chose the
  more expensive option deliberately, having seen others struggle at 256 — ADR-002
- Backgrounds: **~50 self-shot photos + DTD**, with cluttered hard negatives — ADR-004
- **SSIM/MS-SSIM implemented by hand**, validated against skimage — ADR-010
- Bonus: **Tier 1 committed, Tier 2 conditional** — ADR-012
- Implementation agent is **Gemini CLI** (agentic, file + shell access)

**Learned / decided independently:**
- The research report **contradicts a mandatory requirement**: it says to abandon Approach A;
  `[REQ-30]` requires both approaches be implemented and compared. Flagged prominently in
  `GEMINI.md`, `requirements.md`, ADR-007 and Phase 06 — this is the most likely scope failure.
- **The OCR resolution ceiling**, which neither source document notices: an A4 page at 512 px gives
  body text an x-height of ~2.6 px against Tesseract's ~10 px. Naive OCR comparison would measure
  *downsampling*, not enhancement, and could rank the raw input above the model output. Led to
  ADR-011's matched-resolution protocol.
- **Tesseract confidence is miscalibrated for CNN-enhanced images** — it can fall while CER
  improves. Hence CER as the primary OCR metric.
- The **worker-RNG fork trap** would silently collapse dataset variety to 1/N with no visible
  symptom. Called out in `conventions.md` §5 and gated in Phase 03.
- **Calibrating generator ranges to measured real-photo statistics, then widening 1.5–2×** is the
  strongest available sim2real lever. Spec-sanctioned (§1.1, §4.4), but the widening is what stops
  it becoming overfitting to 20 photos — ADR-004 §3.
- Rendering heatmap Gaussians **full-frame would be ~100× slower** than a ±3σ window for an
  identical result — material given Colab's ~2 vCPUs.

**Deliverable:** `.agents/` — 68 documents: the operating contract, requirements and constraints
registers, conventions, deliverables checklist, quick reference, 12 ADRs, 6 research notes, 7 specs,
11 phase files, 6 skills, 3 workflow docs, and the state system with templates.

**Next:** the human chooses the implementation repo location; Gemini starts at
`04-phases/phase-00-foundation.md`. Highest priority on day one is delivering the two capture briefs
to the human — they gate Phase 01 and the ADR-004 calibration.

**Commits:** none — this environment was authored outside a repository.
