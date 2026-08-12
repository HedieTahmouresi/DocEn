# Skill: Portable Training Across Three Machines

**Load when:** moving work between machines, setting up Colab, or debugging an
"it worked on the other machine" problem.

---

## The three environments

| | Workstation | MX330 laptop | Colab T4 |
|---|---|---|---|
| GPU | **none** | 4 GB, Pascal GP108 | 16 GB, Turing |
| Compute | CPU only | ~1.1 TFLOPS FP32 | ~8 TFLOPS FP32 + tensor cores |
| AMP | no | **no** (no usable tensor cores) | **yes** |
| CPU cores | 4 | ~4–8 | **~2** ← the bottleneck |
| RAM | 7 GB | varies | ~12 GB |
| Session limit | none | none | **yes, and it disconnects** |
| Batch @512 | 1–2 | 2–4 | 8–16 |

**The two asymmetries that drive every design choice:**
1. The T4 is ~15–20× faster than the MX330 — so all reported training goes there (ADR-001).
2. Colab has the **fewest CPU cores**, and this project's data pipeline is CPU-bound (ADR-003) — so
   the generator must be fast, not just correct.

## Division of labour

| Work | Where |
|---|---|
| Generator development, QA, visualisation | Workstation |
| Annotation parsing, real-photo prep | Workstation |
| Evaluation, metrics, **all OCR**, plots, report | Workstation |
| Unit tests | Workstation |
| "Does it train at all" smoke tests | MX330 |
| Overnight low-priority runs; Colab-outage fallback | MX330 |
| **Every reported training run** | Colab T4 |

Run **all OCR on one machine** — Tesseract versions differ, and that would silently confound the
comparison (`eval-integrity.md` §2).

---

## Portability rules

**No absolute paths.** Everything resolves against `DATA_ROOT` / `RUNS_ROOT`, from an env var or a
gitignored local `paths.yaml`. Check with:
```
grep -rn "/home/\|/content/\|C:\\\\" src/ configs/ *.py
```
Should return nothing. Worth wiring into a pre-commit check.

**Env profiles carry machine-specific values only:** device, batch size, `num_workers`, AMP. The
experiment config stays identical across machines, so `configs/exp/exp-014.yaml` means the same
experiment everywhere.

```
configs/env/local_cpu.yaml    device: cpu    amp: false  batch: 2   workers: 4
configs/env/mx330.yaml        device: cuda   amp: false  batch: 4   workers: 4
configs/env/colab_t4.yaml     device: cuda   amp: true   batch: 16  workers: 2
```

> **AMP off on the MX330.** Pascal GP108 has no usable FP16 tensor throughput — AMP adds overhead
> for no gain, and can introduce precision problems for nothing.

**Everything must run on CPU**, at reduced size. The generator, the metrics and the evaluation code
are all developed on a machine with no GPU.

**Code by git, data and weights by Drive.** Cloning a repo into Colab is fast and versioned; syncing
a multi-GB dataset through git is not, and one committed checkpoint makes every future clone slow
forever.

---

## Colab specifics

### The notebook is a launcher, nothing more

```
1. mount Drive
2. git clone / git pull
3. pip install -r requirements.txt
4. symlink DATA_ROOT and RUNS_ROOT into Drive
5. !python train.py --config configs/exp/exp-014.yaml
```

**Keep logic out of the notebook.** Notebook code is not meaningfully versioned, cannot be tested,
and cannot be reviewed — and `[REQ-43]` asks for a "modular, executable codebase".

### Surviving disconnection

Sessions die — from timeout, idleness, or nothing in particular.

- **Checkpoint every epoch, to Drive.** Not to the Colab local disk, which vanishes.
- **`--resume` must restore** model, optimizer, scheduler, epoch and RNG state.
- **Test resume before the first long run.** A resume path only exercised after a crash is a resume
  path that does not work.
- **Metrics stream to a file**, not only stdout — a disconnected notebook loses its cell output, and
  with it the entire training history.
- **Prefer shorter epochs.** 4000 samples/epoch gives frequent checkpoints and fine-grained curves;
  a 90-minute epoch loses up to 90 minutes on disconnect.
- Keep runs under ~4 hours where possible.

### Drive is slow

- Do not read the dataset from Drive during training — copy it to the local Colab disk at session
  start. Reading thousands of small files over Drive is far slower than the generator itself.
- Write checkpoints to Drive (they must persist), but keep the write cheap: one file per epoch, not
  per step.

### Quota

Free-tier T4 access is not guaranteed (`[OPEN-08]`). If throttling blocks progress, **escalate** —
options are Colab Pro, Kaggle (30 GPU-h/week and **4 vCPUs**, a better CPU:GPU ratio for this
CPU-bound pipeline), or the MX330 with reduced scope. **Do not silently drop an ablation**; that is
a scope change and the human's call.

---

## MX330 specifics

- **Smoke tests, not training runs.** A few hundred steps to confirm a config works before spending
  Colab time on it.
- 4 GB VRAM: batch 2–4 at 512. If OOM, halve `base_channels` before reducing batch further —
  BatchNorm with batch 1 is unstable.
- AMP off.
- Its advantage is **no session limit**: good for a long, low-priority run you can leave overnight.

---

## Before every cross-machine move

```
[ ] Committed and pushed
[ ] No absolute paths (run the grep)
[ ] Correct env profile selected
[ ] DATA_ROOT / RUNS_ROOT resolve on the target machine
[ ] frozen/ present and loading byte-identically
[ ] Smoke test: a few steps run before launching anything long
```

## "It worked on the other machine"

| Symptom | Likely cause |
|---|---|
| Path not found | Absolute path, or `DATA_ROOT` unset |
| OOM | Wrong env profile (Colab batch size on the MX330) |
| NaN loss appearing only on Colab | AMP — cast MS-SSIM to float32 |
| Much slower than expected | Wrong `num_workers`, or reading data from Drive |
| Frozen-set metrics differ across machines | Frozen sets not actually identical — verify byte-for-byte |
| OCR numbers differ | Different Tesseract version — run all OCR on one machine |
| Different results with the same seed | Expected. Cross-machine bitwise reproduction is not a goal; the **frozen sets** are what make runs comparable |
