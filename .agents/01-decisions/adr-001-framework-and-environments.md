# ADR-001 — Framework, Language, and the Three-Machine Setup

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Low (pervasive)

## Context

Spec §2.2 allows either PyTorch or TensorFlow/Keras. Available compute is unusual and constrains
the design more than the framework choice does:

| Machine | Spec | Role available |
|---|---|---|
| Linux workstation (this box) | 4 cores, 7 GB RAM, **no GPU**, torch 2.13 CPU-only | Development only |
| Laptop with GeForce MX330 | 4 GB VRAM, Pascal GP108, 384 CUDA cores, ~1.1 TFLOPS FP32, **no tensor cores** | Weak training |
| Google Colab, T4 | 16 GB VRAM, ~8 TFLOPS FP32 + FP16 tensor cores, **~2 vCPUs**, session timeouts | Strong training |

The T4 is roughly **15–20× faster** than the MX330 on this workload once mixed precision is used —
and AMP is worthless on the MX330, whose FP16 throughput on GP108 is a fraction of its FP32 rate.
The MX330's 4 GB also caps batch size to ~2–4 at 512×512.

The T4's weakness is the mirror image: ~2 vCPUs, against 4 on the local box. Since the training
data is composited on the CPU on every `__getitem__` (`[REQ-11]`), the T4 is at real risk of
being data-starved rather than compute-bound.

## Decision

**1. PyTorch.**
- Spec §7 names `kornia` for the differentiable warp in the bonus; kornia is PyTorch-only.
- `Dataset`/`DataLoader` with `num_workers` and `worker_init_fn` is the cleanest way to handle the
  CPU-bound generator, and gives explicit per-worker RNG control (`conventions.md` §5).
- The research report and the analysed baseline are both PyTorch.

**2. Colab (T4) is the primary training environment.** The MX330 laptop is for smoke tests and as
a no-timeout fallback. This workstation runs everything that is not training.

**3. The repo must be portable across all three.** This is the binding part of the decision:

- **No absolute paths.** All paths come from config, resolved against a single `DATA_ROOT` /
  `RUNS_ROOT` set per machine (env var or a small local, gitignored `paths.yaml`).
- **Device, batch size, `num_workers`, and AMP are config values**, with a per-machine profile:
  `configs/env/local_cpu.yaml`, `configs/env/mx330.yaml`, `configs/env/colab_t4.yaml`.
  AMP **on** for T4, **off** for MX330 (no tensor cores) and CPU.
- **Checkpoint every epoch** and support `--resume`. Colab sessions die; an un-resumable 6-hour run
  is a 6-hour loss.
- **Metrics stream to a file** (`metrics.json` / CSV) in the run directory, not only to stdout — a
  disconnected Colab notebook loses its cell output.
- **Code moves by `git clone`; data and checkpoints move by Google Drive.** Cloning a repo into
  Colab is fast and versioned; syncing a multi-GB dataset through git is not.
- **Everything must run on CPU too**, at reduced size, so the generator and evaluation code can be
  developed and tested on this workstation without a GPU.

**4. Where each kind of work happens:**

| Work | Machine |
|---|---|
| Generator development, verification, visual QA | workstation |
| Annotation parsing, real-photo prep | workstation |
| Evaluation, metrics, OCR, plots, report | workstation |
| Unit tests, quick sanity runs | workstation |
| "Does this train at all" smoke tests (a few hundred steps) | MX330 |
| Overnight low-priority runs; Colab-outage fallback | MX330 |
| Every reported training run | Colab T4 |

## Consequences

**Good.** The expensive resource (T4 hours) is spent only on training. Development is unblocked by
GPU availability. Runs are reproducible and comparable because config and metrics travel with the
checkpoint.

**Costs.** Extra plumbing up front (env profiles, resume logic, Drive sync) — roughly a day in
Phase 00. Some friction moving between environments.

**Risk — CPU starvation on Colab.** With ~2 vCPUs the generator may not keep a T4 fed. This is the
main technical risk of the compute setup and is addressed directly by ADR-003 (asset caching,
localised heatmap rendering, throughput gate in Phase 02). **It must be measured, not assumed:**
Phase 02's gate includes a samples/s benchmark, and Phase 03's includes a GPU-utilisation check.

**Risk — Colab quota.** Free-tier T4 access is not guaranteed. If throttling becomes a blocker,
escalate: options are Colab Pro (~$10/mo), Kaggle Notebooks (30 GPU-h/week, 4 vCPUs — a better
CPU:GPU ratio for this workload), or falling back to the MX330 with reduced scope. Do not silently
absorb it by shrinking the experiment matrix — that is a scope change and needs the human.

## Alternatives considered

- **TensorFlow/Keras.** Equally permitted, and `tf.data` parallelism is good. Rejected: kornia's
  differentiable warp for the bonus has no clean TF equivalent, and the surrounding
  material assumes PyTorch.
- **MX330 as primary.** Rejected: 15–20× slower, and the ablation matrix (4 loss variants +
  2 corner approaches + dropout variants) would not fit in the schedule.
- **CPU-only training on the workstation.** Not viable at 512×512.

## Validation

- `[ASM-01]` The T4 is ~15–20× faster than the MX330 here. Estimated from FLOPs and tensor-core
  availability, **not measured.** Validate with an identical short benchmark run on both during
  Phase 00. If the gap is much smaller, reconsider the fallback ordering.
