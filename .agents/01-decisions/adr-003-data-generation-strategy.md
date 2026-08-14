# ADR-003 — Data Generation: On-the-Fly Training, Frozen Eval Sets, Cached Assets

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Medium

## Context

The spec already settles the headline question, in two places that must be read together:

- `[REQ-11]` (§2.2): "Your Dataset class should generate samples **on the fly**: each `__getitem__`
  call composites a fresh (degraded input, clean target, corner coordinates) triple."
- `[REQ-15]` (§2.3): "Generate the validation and test samples **once** with a fixed random seed
  (or write them to disk), so that every epoch, and every model you compare, is scored on identical
  images."

So the spec mandates a **hybrid**: fresh samples for training, frozen samples for evaluation. The
question "on-the-fly or pre-generated?" is already answered, per split.

What remains genuinely open is the **engineering** question: on-the-fly compositing is CPU-bound,
and ADR-001 puts primary training on Colab with ~2 vCPUs. If the generator cannot keep the T4 fed,
GPU hours are wasted waiting on `cv2.warpPerspective`.

Rough budget: a T4 running a 4-level U-Net at 512×512 with AMP consumes on the order of 40–80
samples/s. Two vCPUs compositing a full six-step degradation pipeline at 512×512, naively written,
plausibly deliver 10–25 samples/s. **That gap is the design problem.**

## Decision

**1. Training data is generated on the fly.** `[REQ-11]`, non-negotiable. The regularisation value
is real: with every parameter randomised per call, the network effectively never sees the same
image twice.

**2. Validation and test sets are frozen by writing to disk**, not by seeding.

Chosen over the seed-based option the spec also allows, because:
- This project runs on three machines with potentially different PyTorch/NumPy/OpenCV versions.
  Seed-based reproduction of a `num_workers>0` pipeline is fragile across all three axes.
- Frozen sets are then *inspectable* — you can look at exactly what every model was scored on.
- It removes eval-set generation from the training-loop CPU budget entirely.

Format: PNG for images (**never JPEG** — it would apply a second, uncontrolled compression on top
of the pipeline's step 6) plus a JSON sidecar for corners and the sampled degradation parameters.
Stored on Drive, regenerable from a recorded generator seed + git commit.

**3. Decoded source assets are cached in RAM at worker start.**

This is the single highest-value optimisation and it does **not** violate `[REQ-11]`: what is
cached are the *inputs* to compositing (decoded, pre-resized clean scans and background images),
not the composited samples. Every `__getitem__` still performs a fresh warp and a fresh randomised
degradation.

Budget check: 200 scans at 1024 px long side ≈ 600 MB as uint8 — too much for 7 GB shared across
workers. Cache at the working resolution plus headroom instead (see
`03-spec/dataset-and-splits-spec.md` for the exact policy), which brings it to the low hundreds of
MB and is comfortable on both Colab and the workstation.

**4. Heatmap targets are rendered locally, never full-frame.** A 512² Gaussian evaluated over the
whole canvas, four times per sample, is ~1 M exp() calls per corner. Evaluated in a ±3σ window at
σ=8 it is ~2.4 k — roughly **100× cheaper** for a numerically identical result (values beyond 3σ
are <0.02% of peak). Paste the patch into a zero array. See ADR-008.

**5. Throughput is a measured phase gate, not a hope.**
- **Phase 02 gate:** benchmark the generator in isolation; record samples/s at 1, 2 and 4 workers
  on the workstation.
- **Phase 03 gate:** measure end-to-end GPU utilisation on Colab. **If sustained utilisation is
  below ~50%, the pipeline is CPU-bound** — apply the optimisation ladder below before launching
  any long run.

**6. The optimisation ladder** (apply in order; stop when the gate passes):
1. Cache decoded assets (decision 3) — usually sufficient on its own.
2. Localised heatmap rendering (decision 4).
3. Raise `num_workers` to core count, enable `persistent_workers` and `pin_memory`.
4. Pre-resize cached scans so the warp operates on smaller sources.
5. Make the most expensive optional degradations fire probabilistically rather than always
   (still `[REQ-36]`-compliant: presence/absence is itself a randomised parameter).
6. Reduce `samples_per_epoch` and increase epoch count — same total data, more frequent
   checkpoints, no throughput change but better Colab-timeout behaviour.

**7. Only if the ladder fails** does a pre-generated training buffer come into play — and it needs
human approval, because it touches `[REQ-11]`. The compromise form that preserves the requirement:
a large on-disk pool regenerated periodically, with `__getitem__` still applying fresh photometric
degradation on top. Do not reach for this speculatively; measure first.

## Consequences

**Good.** Spec-compliant by construction. Effectively infinite training variety. Evaluation is
stable, inspectable and identical across every compared model and machine. Throughput risk is
detected by a gate rather than discovered after a wasted Colab session.

**Costs.** Frozen sets consume disk and must be regenerated whenever the generator changes in a way
that affects the evaluation distribution — **and that regeneration invalidates comparability with
earlier runs.** Record the generator version (git commit) in the frozen-set manifest, and treat
regeneration as a versioned event in `state/experiments.md`.

**Trap to avoid.** With `num_workers > 0`, workers fork with identical RNG state. Without an
explicit per-worker seed in `worker_init_fn`, every worker composites the *same* samples — the
dataset silently collapses to 1/N of its intended variety while looking perfectly healthy. See
`00-project/conventions.md` §5.

## Sizing

Frozen sets must be large enough that their metrics are not noise. With an 80/10/10 split of ~200
scans, val and test hold ~20 source scans each — too few samples if generated once per scan.
Generate **multiple degradations per held-out scan**: target ~500 frozen validation samples and
~500 frozen test samples (i.e. ~25 degradations per scan). Adjust once the real scan count is known
in Phase 00; record the final numbers in the frozen-set manifest.

## Alternatives considered

- **Fully pre-generated training set.** Decouples CPU from GPU completely and maximises throughput.
  **Rejected: it directly contradicts `[REQ-11]`**, and it fixes the dataset size, reintroducing
  the overfitting risk that on-the-fly generation exists to remove.
- **Seed-frozen eval sets** (the spec's other option). Cheaper in disk, worse across three
  machines, and not inspectable.
- **GPU-side degradation** (kornia). Would solve the bottleneck outright. **Rejected: `[CON-03]`**
  — the degradation pipeline must be OpenCV.
