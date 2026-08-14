# Dataset, Splits and Loaders

**Requirements:** `[REQ-11]` on-the-fly Dataset · `[REQ-12]` scale corners with images ·
`[REQ-13]` normalisation · `[REQ-14]` split by source scan 80/10/10 · `[REQ-15]` freeze val/test ·
`[REQ-16]` real photos as a fourth set · `[REQ-17]` shared split · `[REQ-18]` verify before training
**Decisions:** ADR-003 (generation strategy), ADR-009 (normalisation)

---

## 1. Four data buckets

| Bucket | Source | Generation | Used for |
|---|---|---|---|
| **Train** | 80% of scans | **On the fly**, fresh every call | Training only |
| **Validation** | 10% of scans | **Frozen on disk** | Monitoring, model selection |
| **Test** | 10% of scans | **Frozen on disk** | Final numbers, touched once (`[CON-07]`) |
| **Real** | 10–15+ photos | Never generated | Real-photo evaluation, never training (`[CON-06]`) |

Split by **source scan**, never by generated sample — "two degraded versions of the same page must
never end up on different sides of a split" (`[REQ-14]`). Both tasks share it (`[REQ-17]`).

Assignment is by **hash of filename**, written to `splits.json` and committed. A hash is stable when
scans are added or the directory is re-read; a shuffled index is not, and a silently-changed split
invalidates every prior comparison.

---

## 2. Dataset classes

Four, over one generator.

### `SyntheticTrainDataset` — on the fly

`__getitem__` composites a fresh sample (`[REQ-11]`). Length is `samples_per_epoch` from config, not
the number of scans — "you can control the effective dataset size per epoch" (spec §3.2).

Two output modes over the same generator call, so both tasks are fed by one implementation:
- `task='enhance'` → `(enhance_input, enhance_target)`
- `task='corners'` → `(composite, corner_target)` where the target is 8 normalised coords
  (Approach A) or 4 heatmaps (Approach B)

### `FrozenEvalDataset` — reads from disk

Loads pre-generated PNGs plus a corners JSON. No generation, no randomness. Used for val and test.

### `RealPhotoDataset` — no degradation, ever

Per `[REQ-16]`:
- **For enhancement:** rectify with the annotated corners, resize and normalise **exactly** as
  synthetic inputs; load and resize the reference scan to the same size for comparison.
- **For corners:** the raw photo, resized and normalised, with annotated corners scaled by the same
  factors (`[REQ-12]`).

**Never runs the degradation pipeline** (`[CON-06]`) — "they arrive degraded by reality."

### `BaselineDataset` — the no-model row

Returns `(degraded_input, clean_target)` from the frozen test set so `[REQ-26]`'s baseline can be
computed with the same code path as the model metrics. Small, and it removes a class of
inconsistency between the baseline row and the model rows.

---

## 3. Freezing val and test

`[REQ-15]`. ADR-003 chose **writing to disk** over the spec's seed-based alternative: three machines
with potentially different library versions make seed-based reproduction fragile, and files are
inspectable.

**Format**
```
frozen/{val,test}/
├── images/
│   ├── 00000_composite.png      corner-detector input
│   ├── 00000_enh_input.png      enhancement input
│   └── 00000_enh_target.png     enhancement target
├── corners.json                 {sample_id: [[x,y] × 4]}, absolute px at 512
└── manifest.json                generator config, git commit, seed, counts, timestamp
```

**PNG, never JPEG** — JPEG would apply a second, uncontrolled compression on top of pipeline step ⑥.

**Sizing.** With ~200 scans, val and test hold ~20 scans each — too few if generated once per scan.
Generate ~25 degradations per scan for **~500 frozen samples each**. Re-derive once the real scan
count is known (`[OPEN-01]`).

**The manifest is the comparability contract.** If the generator changes in a way that affects the
evaluation distribution, the frozen sets must be regenerated — and **every earlier run's metrics
become incomparable**. Treat regeneration as a versioned event: record it in `state/experiments.md`,
bump a `frozen_version`, and never mix versions in one table.

---

## 4. Normalisation statistics

ADR-009: per-channel mean/std computed **once**, from **training-split** generated inputs only,
stored in config, never recomputed per run.

Using val or test data here is leakage. A few thousand samples is ample — the estimate converges
quickly.

`[REC]` If the result is near `mean≈0.5, std≈0.25`, using those round numbers is fine and more
robust to a regenerated dataset. Log whichever you use.

---

## 5. DataLoader configuration

```yaml
loader:
  batch_size:         16          # per env profile; 4 on MX330
  num_workers:        2           # Colab ~2 vCPU; 4 on workstation
  persistent_workers: true        # avoids re-warming the asset cache every epoch
  pin_memory:         true        # GPU only
  prefetch_factor:    4
  drop_last:          true        # keeps BatchNorm statistics stable
```

### The worker RNG trap — read this

With `num_workers > 0`, workers **fork with identical RNG state**. Without explicit per-worker
seeding, every worker generates the *same* samples: the dataset silently collapses to 1/N of its
intended variety while looking perfectly healthy — loss decreases, nothing errors.

Use `worker_init_fn` to give each worker its own `np.random.Generator`, seeded from
`base_seed + epoch·num_workers + worker_id`. **Never touch global `np.random` inside a Dataset.**

**Test for it:** pull two batches with `num_workers=4` and assert the samples differ across worker
boundaries.

### Asset cache

ADR-003 decision 3: decode and pre-resize scans and backgrounds into RAM at worker startup. This
does not violate `[REQ-11]` — it caches the *inputs* to compositing, not the samples.

`persistent_workers: true` matters here: without it, the cache is rebuilt every epoch.

**Budget:** with `num_workers=N`, the cache exists **N times** unless it is shared. On Colab (2
workers, ~12 GB) a few hundred MB each is fine. On this workstation (7 GB, 4 workers) check the
arithmetic before caching aggressively — this is a plausible OOM.

---

## 6. Throughput gate

The main risk in ADR-001/003: ~2 vCPUs on Colab may not keep a T4 fed.

| Gate | Phase | Check |
|---|---|---|
| Generator throughput | 02 | samples/s at 1, 2, 4 workers, recorded |
| End-to-end utilisation | 03 | **sustained GPU utilisation ≥ ~50%** on Colab |

Below 50% the pipeline is CPU-bound — apply ADR-003's optimisation ladder before launching any long
run. Do not "just start training and see"; a starved 6-hour run is 6 hours lost.

---

## 7. Pre-training verification — `[REQ-18]`

Before any training, and as a Phase 03 gate:

- [ ] Loader iterates a full epoch without error, in all four bucket types
- [ ] Shapes, dtypes and ranges match `00-project/conventions.md` §3 at every boundary
- [ ] **Visualise (input, target) pairs side by side** (`[REQ-18]`)
- [ ] **Overlay corner labels on composites**, colour-coded per `conventions.md` §8
- [ ] Corners survive resizing correctly (`[REQ-12]`) — round-trip a known coordinate both ways
- [ ] Normalised coords are in `[0,1]` (`[REQ-13]`)
- [ ] **No scan appears in more than one split**; check for near-duplicates too
- [ ] Frozen val/test are byte-identical across two loads and across machines
- [ ] Worker RNG test (§5) passes
- [ ] Real-photo bucket never touches the degradation pipeline (`[CON-06]`) — assert it in code

---

## 8. The `[REQ-26]` baseline, computed from the same path

Spec §3.3: compute the no-model baseline **first** — PSNR/SSIM of the degraded input against the
clean target on the test bucket.

Do it through `BaselineDataset` and the same metric functions as the model rows. Two reasons: it
calibrates the scale (on synthetic document data the degraded-vs-clean PSNR is often already
respectable, so a model at 24 dB may be doing very little), and it catches bugs — if the baseline
*beats* the model, something is inverted, misaligned or in the wrong colour space, and you have
found it before writing the report.

---

## 9. The optional experiment from spec §3.2

> 🧩 "does the model benefit more from seeing many different degradations of few scans, or few
> degradations of many scans?"

Marked **Option**, and easy to run once the loader is config-driven: hold `samples_per_epoch` fixed
and vary the number of distinct source scans sampled from.

`[REC]` Worth doing if Phase 04 has slack — it is a genuine data-centric result and speaks directly
to the generalisation story. Not required. Log it as an experiment and report it if run.
