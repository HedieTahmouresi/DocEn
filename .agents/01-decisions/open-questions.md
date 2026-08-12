# Open Questions

Things that are **not** decided. Check this file at session start (`GEMINI.md` §1). If an open
item blocks your next action, **stop and ask the human** — do not pick an answer and proceed.

Status: `OPEN` · `ANSWERED` (record the answer, then act on it) · `CLOSED` (no longer relevant)

---

## Blocking — needs the human before dependent work can proceed

### `[OPEN-01]` — Contents of the provided clean-scan dataset · **OPEN**
**Blocks:** Phase 00 gate, and therefore all downstream sizing.

The human has the teaching staff's clean scans but they are not on this machine. Unknown until
intake: how many scans, native resolution, DPI, colour vs greyscale, aspect-ratio spread, text
density, file format, whether any are already degraded or skewed.

Everything sized against them is currently provisional: the 80/10/10 split counts (`[REQ-14]`),
the frozen val/test sizes (ADR-003 assumes ~200 scans), the RAM cache budget, and the
many-degradations-of-few-scans experiment in spec §3.2.

**Action:** run the intake audit in `03-spec/data-contract.md` the moment the data lands, write the
inventory into `state/discoveries.md`, and update the affected numbers.
**If the count is far from ~200** (say under 50, or over 1000), re-check ADR-003's sizing and flag
it — a small scan pool materially raises the overfitting risk and changes the split arithmetic.

### `[OPEN-02]` — Real test photos and their annotations · **OPEN**
**Blocks:** Phase 01 gate, the range calibration in ADR-004 §3, and every real-photo evaluation.

Four human-only actions (`00-project/deliverables-checklist.md` §5): capture 10–15 diverse photos
(`[REQ-02]`, `[REQ-04]`), produce a commercial reference scan for each (`[REQ-03]`), annotate four
corners in RoboFlow in TL/TR/BR/BL order (`[REQ-05]`), and upload the project link to the Google
Sheet (`[REQ-06]`).

**`[REC]` — capture 20–25 rather than the required 10–15.** The spec's minimum is 10–15, but this
is the *only* preview of reality and it also feeds the calibration in ADR-004. Extra photos cost
minutes and reduce the noise in every real-photo number reported. Prioritise hard cases: strong
shadow across the page, cluttered background with other papers, oblique angle, dim warm light.

**Note:** the reference scan must be produced *at capture time*, while the document is still in
front of the camera. Reconstructing them later is tedious (spec §1.1 hint).

### `[OPEN-03]` — Background photos · **OPEN**
**Blocks:** Phase 02 (can start with DTD alone, but calibration and the hard-negative strategy in
ADR-004 need the self-shot set).

~50 phone photos of realistic surfaces, of which at least 15 are deliberately cluttered hard
negatives. Full capture brief in ADR-004 §1.

---

## Non-blocking — resolve during the phase that touches them

### `[OPEN-04]` — Does α = 0.84 transfer to document images? · **OPEN**
**Resolve in:** Phase 04. Tracked as `[ASM-04]`.

The value comes from Zhao et al. (2017), tuned on natural-image restoration. Documents are
high-contrast and near-bimodal. Plan: train L-C at α=0.84, then a short sweep over
α ∈ {0.7, 0.84, 0.95}. Flat ranking → keep 0.84 and say so. **No escalation needed** — run it and
report it.

### `[OPEN-05]` — Heatmap σ · **OPEN**
**Resolve in:** Phase 06. Tracked as `[ASM-05]`. ADR-008 sets σ=8 px @512 as a starting point and
prescribes a sweep over {4, 8, 12}. **No escalation needed.**

### `[OPEN-06]` — Is plain MSE sufficient for heatmap training? · **OPEN**
**Resolve in:** Phase 06. ADR-008 documents the foreground/background imbalance (~0.7% positive
pixels), the collapse signature, and a **pre-approved** switch to foreground-weighted MSE.
**No escalation needed for the weighted variant.** Escalate only if you want Adaptive Wing Loss.

### `[OPEN-07]` — Are distractor quadrilaterals needed? · **OPEN**
**Resolve in:** Phase 06. ADR-004 §2 makes synthetic hard-negative distractors a measured ablation
rather than a baseline feature. If corner accuracy on real cluttered photos is already good, skip
them — that is the scope-guard call, and skipping is a reportable finding.

---

## Watch items — may become questions

### `[OPEN-08]` — Colab quota sufficiency · **WATCH**
ADR-001 puts every reported run on a free-tier T4, and ADR-002 chose the ~4× more expensive
resolution. Free-tier access is not guaranteed.

**If throttling blocks progress, escalate — do not silently shrink the experiment matrix.**
Dropping an ablation is a scope change and is the human's call. Options to present: Colab Pro
(~$10/mo), Kaggle Notebooks (30 GPU-h/week, 4 vCPUs — better CPU:GPU ratio for this CPU-bound
pipeline), MX330 fallback with reduced scope, or the resolution ladder in ADR-002.

Bring measured evidence: seconds/epoch, GPU utilisation, projected total hours.

### `[OPEN-09]` — Is BatchNorm acceptable under `[CON-04]`? · **WATCH**
ADR-005 records the reading — BatchNorm is a normalisation/optimisation layer, not an *explicit*
regulariser like dropout or weight decay — and this is the standard interpretation. Low risk, but
it depends on a TA's judgment.

**Contingency, not a task:** if ruled otherwise, swap to GroupNorm or InstanceNorm. Localised
change, but it invalidates trained checkpoints, so ask early if there is any chance of clarifying
with a TA.

### `[OPEN-10]` — Does the enhancement model beat the raw photo on OCR? · **WATCH**
ADR-011 flags the risk that a 512-bottlenecked output OCRs *worse* than the full-resolution
rectified input, purely because of downsampling. The matched-resolution protocol removes this as a
confound from the main comparison, and the full-resolution row is reported separately as an honest
disclosure.

**If the full-resolution raw input wins, report it.** It is a real limitation of the resolution
choice and belongs in `[REQ-48]`. It is not a reason to change the protocol after seeing the
result — see `05-skills/eval-integrity.md`.

---

## How to add an item

Append with the next `[OPEN-nn]` number and record: what is unknown, what it blocks, what would
resolve it, and whether it needs the human or just needs doing. When answered, change the status to
`ANSWERED`, record the answer inline, and — if it settles something durable — write or update an
ADR. Do not delete answered items; the trail is the point.
