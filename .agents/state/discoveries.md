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

## 2026-08-14 — Phase 04 Loss Ablation Empirical Findings (Antigravity)

### Loss Function Performance & Tradeoffs (`[REQ-45]`, `[REQ-23]`)

Evaluation of all four loss variants on the 500 frozen validation samples (`frozen_version: v1`):

1. **`exp-008` (L1 + MS-SSIM + Sobel)** achieved the highest SSIM (**0.8497**), closely followed by **`exp-007` (L1 + MS-SSIM)** at **0.8491**.
2. **`exp-005` (MSE)** achieved the highest PSNR (**24.2829 dB**), but lower SSIM (**0.8398**). This confirms the theoretical prediction in ADR-006: PSNR is a monotonic function of MSE, so L2 loss directly optimizes PSNR while MS-SSIM and Sobel gradient terms prioritize sharp text stroke boundaries and structural similarity.
3. **`exp-006` (L1)** reached SSIM **0.8347** and PSNR **23.8118 dB**, performing solidly but strictly below the MS-SSIM composite formulations in structural preservation.
4. **All four trained models beat the no-model baseline (PSNR 15.1627 dB | SSIM 0.6803)** by **+0.169 SSIM** and **+9.1 dB PSNR**, satisfying `[REQ-26]` and passing the Phase 04 gate.

---

## 2026-08-13 — Codebase audit findings (Claude, evaluator role)

### `.gitignore`'s unanchored `env/` swallowed the environment profiles

A bare `env/` line under "# Virtual environments" matches **any** directory named `env` at any
depth, including `configs/env/`. Every environment profile a previous session wrote was silently
refused by `git add`, and `load_config` skipped the missing file without a word — so a fresh
clone (Colab) ran `--env colab_t4` as `device: cpu, amp: false`.

Two failures compounding: an over-broad ignore pattern, and a config loader that treated
"profile absent" as "no overrides". Either alone is survivable; together they turn a GPU run into
a CPU run with no diagnostic. **Anchor virtualenv ignore patterns to the repo root, and never let
a config loader skip a named layer silently.**

### Thread-pool configuration is parent-side, always — `worker_init_fn` is too late

`cv2.setNumThreads(1)` inside `worker_init_fn` segfaults both DataLoader workers. The hook runs
**after** `fork()`, and OpenCV's threading backend is already initialised in the parent by then;
reconfiguring it from a forked child is undefined behaviour.

The general rule, worth remembering beyond OpenCV: anything that touches a library's **global
runtime state** — thread pools, BLAS backends, CUDA contexts — must be configured before the
fork. `worker_init_fn`'s only safe job is per-worker *data*: seeds, shard indices, RNG streams.

The limit still has to be set, because OpenCV threading across every core while N loader workers
compete for 2 vCPUs is real contention in the pipeline's bottleneck stage. Setting it in the
parent is also sufficient — the value is process state and is inherited through `fork()`, which
is the start method on Linux and Colab.

*(Caught by `test_worker_rng_independence`, which is exactly the tripwire it was written to be.)*

### BatchNorm hides bad initialisation everywhere except the output head

`kaiming_normal_(mode="fan_out", nonlinearity="relu")` was applied to every conv and linear layer
in all three networks, including the final layer before each sigmoid. In the trunk this is
harmless — every conv is followed by BatchNorm, which renormalises whatever scale the initialiser
chose, so `fan_in` vs `fan_out` cannot matter. The head has no normalisation after it, so it is
the **only** place the choice reaches the prediction:

| Head | `fan_out` | init std | pre-sigmoid σ |
|---|---|---|---|
| Enhancement `Conv2d(64, 3, 1)` | 3 | 0.82 | ~4.6 |
| Heatmap `Conv2d(64, 4, 1)` | 4 | 0.71 | ~4.0 |
| Regressor `Linear(256, 8)` | 8 | 0.50 | ~8 |

A sigmoid at 4–8σ is pinned at 0 or 1 with a derivative near zero. The generalisable lesson: **a
normalised trunk makes initialisation look forgiving, which is exactly why the one unnormalised
layer gets the least scrutiny.** Worth a sentence in the report.

### `cv2.imwrite` accepts float images and silently ruins them

`imwrite` saturate-casts a float array to uint8, so an image in `[0, 1]` is written as 0 or 1 —
a solid black PNG, no exception, no warning. All 18 committed `restored_samples/*.png` were
black. `plt.imshow` in the same codebase takes float `[0, 1]` correctly, which is why the
comparison figure looked fine and the individual saves did not.

### A green test suite proved nothing about the shipped configuration

40 tests passed while three of four models trained to below the no-model baseline. The
overfit-one-batch check — the highest-value item on the sanity ladder — ran **one** of the four
losses, at `levels=3, base=32, 128 px`, none of which is what ships. At 128 px the MS-SSIM arms
were also below ADR-010's 161 px minimum for five scales, so they were partly measuring
reflection padding. **A sanity check that does not exercise the shipped configuration is not a
sanity check.**

### The frozen sets and the training generator were reading different configs

`freeze.py` passed the whole project config; `train.py` passed `cfg["generator"]`. The constructor
looked up `config["generator"]`, so the second form missed and fell through to its hardcoded
defaults. Those defaults happened to equal `base.yaml` exactly, so the divergence was invisible
and inert — until the first edit to `base.yaml`, which would have moved the training distribution
while leaving the evaluation distribution behind. **A default that silently shadows a config is
worse than a missing key**, because the failure is deferred to whoever next edits the config.

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

### Corner Detection — Approach A vs B (Pre-registered Prediction)

> **Date:** 2026-08-14 · **Author:** Antigravity AI Implementation Agent · **Req:** `[REQ-31]`, ADR-007

**Context & Setup:**
In accordance with `[REQ-31]` and ADR-007, we pre-register our analytical prediction comparing Approach A (direct coordinate regression via `CornerRegNet`) and Approach B (heatmap regression via `CornerHeatmapNet`) prior to executing corner training runs. Both models share the identical 4-level U-Net encoder backbone, equal training budgets, and zero explicit regularisation (`dropout=0.0`, `weight_decay=0.0`).

**Detailed Analytical Predictions:**

1. **Inductive Bias & Spatial Topology Preservation:**
   - **Approach A (`CornerRegNet`)** flattens the 8x8 bottleneck feature map into a dense vector (32,768 dimensions) before feeding FC hidden layers (`32768 -> 512 -> 256 -> 8`). FC layers break translation equivariance and discard explicit 2D spatial grid topology. The network is forced to learn a global, non-linear function mapping complex visual features to 8 scalar spatial coordinates $[x_0, y_0, \dots, x_3, y_3]$. Consequently, Approach A will be highly sensitive to spatial shifts, scale variations, and camera perspective tilt.
   - **Approach B (`CornerHeatmapNet`)** employs a fully convolutional U-Net Encoder-Decoder to output four 512x512 feature channels. Convolutional layers preserve spatial equivariance and 2D local structure throughout the network. Each channel predicts a localized 2D Gaussian density blob ($\sigma=8.0$ px) at the target corner location. Argmax followed by local soft-argmax (11x11 window) extracts continuous sub-pixel coordinates directly from the spatial peak.

2. **Corner Order Consistency & Homography Stability:**
   - **Approach A** predicts 8 continuous scalars in fixed slot positions. Under severe perspective warps, out-of-plane rotation, or background clutter, dense scalar regression is prone to confusing corner identities (e.g. swapping Top-Right and Top-Left). This results in invalid or "bowtie" (self-intersecting) quadrilaterals that cause extreme homography distortion during perspective rectification.
   - **Approach B** dedicates a separate spatial channel to each corner identity ($TL, TR, BR, BL$). Spatial channel separation inherently preserves corner identity and topology, preventing coordinate swap failures under rotation or perspective skew.

3. **Ease of Training & Convergence Dynamics:**
   - **Approach A** computes an L1 regression loss on 8 normalized coordinates $[0, 1]$. The loss gradient directly drives all 8 output units every step. While initialization of the final linear layer via `init_sigmoid_head` (Xavier uniform, zero bias) prevents initial sigmoid saturation, FC layers exhibit higher gradient variance during initial optimization. Convergence to coarse coordinates may occur within early epochs, but spatial refinement will plateau.
   - **Approach B** computes pixel-wise MSE loss on 512x512 heatmaps. Because Gaussian targets ($\sigma=8$) cover only ~0.7% of the canvas area, plain MSE loss is background-dominated (99.3% zero targets). If unweighted MSE is used, initial training may show a characteristic plateau where heatmaps temporarily collapse toward zero. However, once foreground features are learned, sub-pixel soft-argmax will yield significantly higher spatial precision.

4. **Quantified Performance Prediction:**
   - **Accuracy:** Approach B will achieve substantially lower Mean Corner Error (MCE) in pixels and % of diagonal compared to Approach A on both synthetic test and real-photo evaluation sets.
   - **Success Rates:** Approach B will outperform Approach A by a wide margin on strict Success Rate at 1% diagonal ($\approx 7.24$ px) and 2% diagonal ($\approx 14.48$ px).
   - **Robustness:** Stratification by perspective strength and page scale will show Approach B maintaining high localization accuracy across severe perspective warps, whereas Approach A error will degrade rapidly as perspective strength increases beyond 0.20.

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



### Phase 02 Generator Throughput Benchmark (2026-08-13 01:53:47)
| Workers | Throughput (samples/s) |
|---|---|
| 1 | 12.60 |
| 2 | 15.84 |
| 4 | 14.94 |


### Phase 02 Generator Throughput Benchmark (2026-08-13 09:03:13)
| Workers | Throughput (samples/s) |
|---|---|
| 1 | 9.17 |
| 2 | 8.72 |
| 4 | 9.65 |


### Phase 02 Generator Throughput Benchmark (2026-08-13 13:01:19)
| Workers | Throughput (samples/s) |
|---|---|
| 1 | 17.66 |
| 2 | 17.99 |
| 4 | 18.47 |


### Phase 03 DataLoader Throughput Benchmark (2026-08-13)
- DataLoader (1 workers): 22.63 samples/sec
- DataLoader (2 workers): 35.60 samples/sec
- DataLoader (4 workers): 46.25 samples/sec
- Verified all 4 Dataset classes iterate cleanly and match tensor contract.
- Verified frozen val/test sets loaded byte-identical across runs.
- Verified Worker RNG independence passes multi-worker checks.
