# Project Conventions

Fixed, project-wide conventions. Every one of these exists because getting it wrong produces a
**silent** bug — training proceeds, loss decreases, and the result is quietly wrong.

Treat this file as binding. If you need to depart from a convention, that is a deviation
(`GEMINI.md` §5).

---

## 1. Corner ordering — the highest-risk convention in the project

**Order is `[top-left, top-right, bottom-right, bottom-left]` — clockwise, starting top-left.**

Index → corner:

```
  0 ────────── 1        0 = TL   top-left
  │            │        1 = TR   top-right
  │            │        2 = BR   bottom-right
  │            │        3 = BL   bottom-left
  3 ────────── 2
```

This comes from `[REQ-05]` (spec §1.2) and must be identical in **all four** places:

1. RoboFlow keypoint annotation of the real photos.
2. The synthetic generator's recorded target points.
3. Heatmap channel index (channel `c` ↔ corner `c`).
4. Approach A's 8-vector: `[x0,y0, x1,y1, x2,y2, x3,y3]`.

**"Top-left" means top-left of the *page*, not of the image.** For a page rotated 40°, TL is the
corner that is top-left in the document's own frame — the corner adjacent to the start of the
text. This matters: if you define it as "whichever predicted point has the smallest x+y", a rotated
page will relabel its corners and the homography will rotate the output.

> ⚠️ **The failure mode.** Wrong ordering does not crash. It produces a homography that flips or
> rotates the page — spec §7 hint calls this out explicitly. Downstream, PSNR/SSIM collapse and it
> looks like a model problem. **Always visually verify ordering** with a colour-coded overlay
> (see §8 below) before trusting any corner-related number.

**Do not "sort" predicted corners at inference** to enforce ordering. Both networks are trained to
produce a specific corner in a specific slot; sorting hides ordering errors instead of revealing
them, and breaks on rotated pages. If ordering is wrong, that is a real error the metrics should
capture.

---

## 2. Coordinate convention

- A point is `(x, y)` where **`x` is the column** and **`y` is the row**. Origin is the top-left
  pixel of the image, `x` increases rightward, `y` increases downward.
- This matches OpenCV (`cv2.circle(img, (x, y), ...)`) and is the **opposite** of NumPy indexing
  (`img[y, x]`). Every conversion between the two is a bug opportunity — name variables `xy` or
  `rc` explicitly rather than `pt`.
- Corner arrays have shape `(4, 2)` in `(x, y)` order, `dtype=np.float32`
  (`cv2.getPerspectiveTransform` requires float32 and will raise otherwise).
- **Normalised coordinates** (`[REQ-13]`) are `x/W`, `y/H`, in `[0, 1]`, where `W`/`H` are the
  dimensions of the image the coordinates belong to *at that moment*. Denormalise against the same
  dimensions.
- Sub-pixel coordinates are expected and fine — do not round until rendering.

**The scaling rule** (`[REQ-12]`): resizing an image without applying the same scale factors to its
corners produces a wrong label, silently. Keep image and corners together in a single object so
they cannot drift apart.

---

## 3. Colour space and dtype at each boundary

| Stage | Layout | dtype | Range | Colour |
|---|---|---|---|---|
| On disk / `cv2.imread` | `HWC` | `uint8` | 0–255 | **BGR** |
| Inside the degradation pipeline | `HWC` | `uint8` (or `float32` for intermediate maths) | 0–255 | **BGR** |
| Handed to the Dataset's tensor conversion | `HWC` | `uint8` | 0–255 | **converted to RGB here** |
| Model input / target | `CHW` (batched `NCHW`) | `float32` | see §4 | RGB |
| Metric computation | `NCHW` | `float32` | `[0, 1]` | RGB |
| Saved output / `cv2.imwrite` | `HWC` | `uint8` | 0–255 | **converted back to BGR** |

**Rules:**
- Stay in BGR for the whole OpenCV degradation pipeline. Convert **once**, at the boundary where
  the numpy array becomes a tensor.
- Convert back to BGR **once**, immediately before `cv2.imwrite`.
- `matplotlib.pyplot.imshow` expects RGB. Displaying a BGR array shows a blue-tinted image — a
  useful canary that you missed a conversion.
- Do the colour-cast degradation (`[REQ-34]` step 3) in BGR and be explicit about which channel
  index is red. In BGR, **index 0 is blue and index 2 is red.** A warm cast boosts index 2.

---

## 4. Normalisation and value ranges

Fixed by ADR-009. Summary:

- **Model input:** standardised with dataset-derived per-channel mean/std, computed once in
  Phase 03 from the training split and stored in the config. **Not** ImageNet constants — there is
  no pretrained network to match, and `[CON-02]` rules them out.
- **Enhancement target:** plain `[0, 1]` (`/255.0`). Not standardised.
- **Enhancement output:** `[0, 1]` via a final `sigmoid`.
- **All metrics (PSNR, SSIM, MS-SSIM):** computed in `[0, 1]` space with `data_range=1.0`.
- **Heatmap targets:** `[0, 1]`, peak value exactly 1.0 at the corner.
- **Approach A targets:** normalised coordinates in `[0, 1]`, output through a `sigmoid`.

Never compute a metric on standardised tensors. Un-standardise, or better, keep the target and
prediction in `[0, 1]` throughout and standardise only the input.

---

## 5. Randomness and determinism

- One global `seed` in the config. Derive everything from it.
- **Frozen val/test sets** (`[REQ-15]`) are generated once and **written to disk**, not
  regenerated from a seed at load time. Reason: with `num_workers > 0`, per-worker seeding makes
  seed-based reproduction fragile across machines and PyTorch versions, and this project runs on
  three different machines. Disk is the only reliable freeze.
- Training-time generation is *deliberately* non-deterministic — that is the point of `[REQ-11]`.
  Seed the worker RNGs from `base_seed + epoch * num_workers + worker_id` so runs are reproducible
  in principle without repeating the same sample stream every epoch.
- **Never use the global `np.random` inside a Dataset worker.** Each worker forks with the same
  state; without per-worker seeding, all workers generate identical samples. Use an explicit
  `np.random.Generator` created in `worker_init_fn`. This is a classic, silent, throughput-wasting
  bug.
- Record the seed in every experiment entry.

---

## 6. Naming

| Thing | Convention | Example |
|---|---|---|
| Experiment ID | `exp-NNN` zero-padded, monotonic, never reused | `exp-014` |
| Run directory | `runs/<exp-id>_<short-slug>/` | `runs/exp-014_enh-l1msssim/` |
| Checkpoint | `epoch_NNN.pt`, plus `best.pt` and `last.pt` symlinks/copies | `epoch_042.pt` |
| Config | `config.yaml` inside the run directory — the **exact** config used | |
| Metrics | `metrics.json` inside the run directory | |
| ADR | `adr-NNN-short-slug.md` | `adr-006-enhancement-loss.md` |
| Branch | `phase/NN-slug` | `phase/02-generator` |
| Figure | `<phase>_<subject>.png` | `p05_triplets_real.png` |

**Variable naming for the three networks** — keep these distinct everywhere, including in logs:
`enhance` (Task 1), `corner_reg` (Approach A), `corner_hm` (Approach B).

---

## 7. Units in reported numbers

Always state the unit and the resolution.

- **Corner error in pixels** is meaningless without a resolution. Report it **at 512×512** (the
  working resolution, ADR-002) and *also* as a fraction of the image diagonal, so the number
  survives any future resolution change and is comparable across the literature.
  512×512 diagonal = 724.1 px.
- **PSNR** in dB, `data_range=1.0`, averaged **per image** then across images — never computed on a
  concatenated batch (that gives a different, wrong number).
- **SSIM** with the settings pinned in `03-spec/evaluation-spec.md`. Report the settings.
- **Success rate** requires the threshold in the same sentence: "success = all four corners within
  T px at 512×512."

Do not compare absolute pixel numbers to the baseline notebook analysed in
`02-research/baseline-failure-analysis.md` — its resolution and threshold are not documented.
Its *pattern* (96% synthetic → 0% real) is the transferable finding, not its magnitudes.

---

## 8. The standard visual check

Any time corners are involved, render this before trusting a number:

- Corner 0 (TL) — **red**, corner 1 (TR) — **green**, corner 2 (BR) — **blue**, corner 3 (BL) —
  **yellow**. Fixed colours, always the same, so a mis-ordering is visible at a glance.
- Draw the quadrilateral edges in order 0→1→2→3→0. A bowtie shape means the ordering is wrong.
- For predictions, draw ground truth as hollow circles and predictions as filled ones on the same
  image.

Store these under `outputs/figures/` and reference them from the session log.

---

## 9. Config over constants

Anything a reasonable person might want to change — resolution, batch size, learning rate,
degradation parameter ranges, loss weights, σ for heatmaps, number of workers — lives in a YAML
config, not in the source. Every run stores its resolved config alongside its metrics
(`06-workflow/git-workflow.md`, `05-skills/experiment-discipline.md`).

Rationale beyond tidiness: `[REQ-43]` requires being able to "adjust hyperparameters, change the
model architecture, add a new degradation" live at the presentation. A config-driven codebase makes
that a 10-second demonstration instead of a code hunt.

---

## 10. Documentation strings

Every function that touches images or coordinates states, in its docstring, the **shape, dtype,
range and colour space** of its inputs and outputs. Example contract style:

```
img:     (H, W, 3) uint8, BGR, 0-255
corners: (4, 2) float32, (x, y), absolute pixels in img's frame, order TL,TR,BR,BL
returns: (H, W, 3) uint8, BGR, 0-255
```

This single habit prevents most of the bugs this file exists to warn about.
