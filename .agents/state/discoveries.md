# Discoveries

Things learned that **change our understanding** — not a work journal (that is `session-log.md`) and
not results (those are `experiments.md`).

**What belongs here:**
- Facts about the data that were unknown or assumed
- Findings that invalidate or modify a plan, decision or assumption
- Non-obvious behaviour discovered while debugging, that a future agent would otherwise rediscover
- **Pre-registered predictions**, dated — spec §5.1 requires one for the corner comparison
- Measurements that other work depends on (throughput, timings, distributions)

**What does not:** what you did today, routine results, anything already written in a spec.

Newest first.

---

## 2026-08-12 — Environment design findings (planning agent)

### The research report contradicts a mandatory requirement

`Document Scanner Implementation Plan.md` instructs "strictly abandoning the direct regression
methodology" for corner detection. `[REQ-30]` (spec §5) requires implementing **both** approaches
and letting experiments decide.

The report is arguing about which model to *deploy*; the spec is requiring both to be *built and
measured*. An agent following the report would drop half a mandatory deliverable.

**Assessed as the single most likely scope failure in the project.** Flagged in `GEMINI.md` §7,
`requirements.md` `[REQ-30]`, ADR-007, and as a Phase 06 gate item.

### The OCR resolution ceiling — noticed by neither source document

An A4 page (297 mm) with 11 pt body text (x-height ≈ 1.5 mm) rendered at 512 px height gives an
x-height of **~2.6 px**. Tesseract wants ~10 px. At 256 it is ~1.3 px.

**512 does not make OCR easy — it makes it less impossible.**

The consequence is subtler than the limitation itself. If the three images in `[REQ-27]`'s
comparison are OCR'd at their *natural* resolutions, the rectified input has full camera resolution
(sharp text, bad lighting), the model output has passed a 512 bottleneck (soft text, good lighting),
and the reference scan is full-resolution and app-processed. Tesseract would then be measuring
**downsampling**, not enhancement — and could plausibly rank the raw input above the enhanced
output.

→ ADR-011's matched-resolution protocol. → Required entry in the `[REQ-48]` limitations.
*(Derived — the arithmetic is ours; check it if it matters.)*

### Tesseract confidence is miscalibrated for CNN-enhanced images

Confidence scores are calibrated against the image statistics Tesseract was trained on. Enhanced
images have altered intensity distributions and can score **lower confidence despite lower CER**.

Confidence and accuracy can move in opposite directions. → ADR-011 leads with CER; confidence is
secondary with the caveat stated. *(Secondary source — see `02-research/source-index.md`.)*

### The worker-RNG fork trap

With `num_workers > 0`, PyTorch DataLoader workers fork with **identical RNG state**. Without
explicit per-worker seeding, every worker composites the *same* samples — dataset variety silently
collapses to 1/N. Nothing errors, loss decreases, training looks healthy, and the model overfits far
faster than it should.

→ `00-project/conventions.md` §5; explicit test as a Phase 03 gate item.

### Full-frame Gaussian rendering is ~100× more work than necessary

A σ=8 Gaussian over a full 512² canvas is ~262 k evaluations per corner per sample. In a ±3σ window
it is ~2.4 k, and values beyond 3σ are under 0.02% of peak — numerically identical result.

Material rather than cosmetic: with four corners per sample on Colab's ~2 vCPUs, the naive version
alone could starve the T4. → ADR-008; Phase 02 gate.

### MS-SSIM has a hard minimum image size

Five scales with an 11×11 window requires ≥ `(11−1)·2⁴ + 1 = 161` px. 512 is comfortable. Relevant
only if anything ever runs at reduced size — reduce the number of scales rather than letting it
throw. *(Verified.)*

### Zhao et al.'s mixed loss is not what most reimplementations use

The paper (arXiv:1511.08861) applies a **Gaussian weighting** to the L1 term:
`L_mix = α·L_MS-SSIM + (1−α)·G_σM·L1`. Most public implementations drop `G_σM`.

The simplified form is acceptable — but read §5 and state which was implemented. Citing the paper
for a formulation you did not implement is the kind of thing that gets caught in a viva. → ADR-006.

### Widening generator ranges beyond observed reality is the point, not a hedge

The instinct when calibrating is to *match* the measured real-photo distribution. That is subtly
wrong: matched ranges put half of real samples in the harder half of the training distribution, and
leave the tails — where failures live — unrepresented.

Domain-randomisation results favour ranges **wider** than the target domain, so the real
distribution is an easy interior case. → ADR-004 §3, ~1.5–2× the observed spread.

The counter-pressure is `[REQ-37]`: "be cautious of excessive degradation". The worst-10 readability
check is the limit.

---

## Pre-registered predictions

Spec §5.1 hint requires this for the corner comparison, and it must be written **before** training.

### Corner detection — Approach A vs B

> **To be completed by the implementation agent in Phase 06, before any corner training run.
> Date it. Then report whether it held.**

Prior evidence to reason from (all in `02-research/corner-localization.md`): fully connected layers
destroy the spatial topology that convolution preserves, converting a spatial task into a global
non-spatial mapping; the analysed baseline scored 8.00% (A) vs 96.00% (B) on the same synthetic
data — though that 96% was itself inflated by a narrow generator.

### Enhancement loss ablation

Recorded in ADR-006 and `experiments.md`: L-C (L1+MS-SSIM) wins on SSIM and sharpness; L-A (MSE)
wins or ties on PSNR while looking worse; L-D (+Sobel) roughly neutral.

---

## Data facts

*Empty until the Phase 00 intake audit. **Several numbers across this environment assume ~200
scans** — split sizes, frozen-set sizing, RAM cache budget. Record the real figures here and update
what depends on them.*

| Fact | Value | Recorded |
|---|---|---|
| Clean scan count | **50** | 2026-08-12 (Phase 00 Audit) |
| Scan resolution (min/median/max) | **(2384×3396) / (2468.5×3512) / (2480×3648)** | 2026-08-12 (Phase 00 Audit) |
| Aspect-ratio distribution | **0.67 – 0.73 (median 0.70)** | 2026-08-12 (Phase 00 Audit) |
| Colour vs greyscale | **50 RGB (3-channel)** | 2026-08-12 (Phase 00 Audit) |
| Split counts (80/10/10) | **41 Train / 5 Val / 4 Test** | 2026-08-12 (Phase 00 Audit) |
| Real photos & reference scans | **30 active photos / 30 reference scans** (5 removed) | 2026-08-12 (Phase 00 Audit) |
| Backgrounds count | **64 files** (1.jpg – 64.jpg, min side 300px) | 2026-08-12 (Phase 00 Audit) |
| Workstation CPU step time | **1058.16 ms/step (1.058 s/step)** @ batch=2 | 2026-08-12 (Phase 00 Audit) |
| Generator throughput (1/2/4 workers) | *unknown* | Phase 02 |
| Sustained Colab GPU utilisation | *unknown* | Phase 03 |
| MX330 vs T4 speed ratio | *estimated 15–20×, unmeasured* | `[ASM-01]`, Phase 00 |
| Normalisation mean/std | *unknown* | Phase 03 |

