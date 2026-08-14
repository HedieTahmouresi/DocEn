# Repository Layout

A target structure, not a straitjacket. The **bolded** items are fixed by the spec or by
cross-references elsewhere in this environment; everything else is ordinary engineering judgment —
reorganise if you have a reason, and note it in the session log.

---

## Structure

```
<repo-root>/
├── GEMINI.md                    copy/symlink of .agents/GEMINI.md — auto-loaded
├── README.md                    how to run each entry point (REQ-43)
├── requirements.txt
├── .gitignore
│
├── .agents/                     this environment — committed, versioned
│
├── model.py                     ★ REQ-20: all architectures
├── train.py                     ★ REQ-21: training entry point
├── evaluate.py                  ★ REQ-24: evaluation entry point
│
├── src/
│   ├── data/
│   │   ├── generator.py         the OpenCV degradation pipeline (CON-03)
│   │   ├── datasets.py          the four Dataset classes
│   │   ├── splits.py            hash-based split assignment
│   │   ├── freeze.py            writes frozen val/test to disk
│   │   └── annotations.py       COCO keypoint parsing (REQ-10)
│   ├── losses/
│   │   ├── ssim.py              own SSIM + MS-SSIM (ADR-010)
│   │   ├── sobel.py
│   │   └── composite.py         the four loss variants
│   ├── metrics/
│   │   ├── image.py             PSNR, SSIM
│   │   ├── corners.py           mean error, success rate, quad IoU
│   │   └── ocr.py               CER, confidence, matched-resolution protocol
│   ├── pipeline/
│   │   ├── enhance.py           ★ REQ-29
│   │   ├── corners.py           ★ REQ-32
│   │   └── scanner.py           ★ REQ-40 (bonus)
│   ├── geometry/
│   │   ├── homography.py        corner ordering, validation, warping
│   │   └── warp.py              cv2 / kornia backends behind one interface (ADR-012)
│   └── utils/
│       ├── config.py            YAML loading, env profile merging
│       ├── seeding.py           global + per-worker RNG (conventions §5)
│       ├── viz.py               the standard colour-coded overlays (conventions §8)
│       └── io.py                image read/write with the BGR/RGB boundary in one place
│
├── configs/
│   ├── base.yaml
│   ├── env/                     local_cpu.yaml · mx330.yaml · colab_t4.yaml
│   └── exp/                     one per experiment; committed
│
├── tests/
│   ├── test_ssim.py             ★ validation vs skimage (ADR-010) — a Phase 04 gate
│   ├── test_generator.py        round-trip alignment, corner ordering, degeneracy
│   ├── test_geometry.py         coordinate scaling round-trip, ordering
│   └── test_datasets.py         worker RNG independence, split disjointness
│
├── notebooks/
│   └── colab_train.ipynb        clone repo, mount Drive, run train.py (ADR-001)
│
├── outputs/                     gitignored except small figures
│   ├── figures/
│   └── reports/
│
└── runs/                        gitignored; configs + metrics copied back to git
```

---

## Why `model.py` / `train.py` / `evaluate.py` sit at the root

The spec names all three explicitly (`[REQ-20]`, `[REQ-21]`, `[REQ-24]`). A grader looking for
`model.py` should find it immediately. Keep them at the root as thin, readable entry points, with
the substance in `src/`.

`model.py` holds all three architectures plus the shared encoder/decoder — `[REQ-20]` says "this
entire architecture will be implemented … in the `model.py` file."

---

## What is and is not committed

**Committed:** all code, `configs/`, `splits.json`, `.agents/`, `metrics.json` and `config.yaml`
copied back from each run, small figures used in the report, `tests/`.

**Gitignored:** `data/` (any location), `runs/*/checkpoints/`, `frozen/`, large figures, `*.pt`,
`*.pth`, `__pycache__`, `.ipynb_checkpoints`, and any local `paths.yaml`.

Rationale in `06-workflow/git-workflow.md`. The short version: **code travels by git, data and
weights travel by Drive** (ADR-001). Committing a checkpoint once makes every later clone slow
forever.

---

## Entry points

```
python train.py    --config configs/exp/exp-014.yaml [--resume runs/.../last.pt]
python evaluate.py --run runs/exp-014_enh-l1msssim [--split test]

python -m src.data.freeze --config configs/base.yaml      # generate frozen val/test
python -m src.pipeline.enhance --image path/to/rectified.jpg
python -m src.pipeline.corners --image path/to/photo.jpg
python -m src.pipeline.scanner --image path/to/photo.jpg  # bonus
```

The three pipeline entry points must be runnable on an arbitrary unseen image
(`[REQ-29]`, `[REQ-32]`, `[REQ-46]`). At the presentation the TAs will run them on photos you have
never seen (`[REQ-49]`) — **they must not crash on an unusual aspect ratio, a greyscale JPEG, or an
image with EXIF rotation.** Test those three cases explicitly before Phase 10 closes.

---

## Config layering

`base.yaml` → `env/<machine>.yaml` → `exp/<experiment>.yaml`, later overriding earlier. The env
layer carries only device, batch size, workers and AMP (ADR-001), so the same experiment config runs
unchanged on all three machines.

Write the **fully resolved** config into the run directory. A config that depends on which files
were merged is not reproducible six weeks later.

---

## Colab notebook

One notebook, deliberately thin (ADR-001):

1. Mount Drive
2. `git clone` (or `git pull`) the repo
3. `pip install -r requirements.txt`
4. Symlink `DATA_ROOT` and `RUNS_ROOT` into Drive
5. `!python train.py --config ...`

**Keep logic out of the notebook.** Code in a notebook is not versioned meaningfully, cannot be
tested, and cannot be reviewed — and `[REQ-43]` asks for a "modular, executable codebase". The
notebook is a launcher.

---

## Dependencies

Keep the list short and justify anything unusual against `[CON-01]`–`[CON-03]`.

```
torch, torchvision        # torchvision for I/O utilities only — NOT models, NOT transforms
                          #   for the degradation pipeline (CON-01, CON-03)
numpy
opencv-python             # the degradation pipeline (CON-03 requires this)
pyyaml
matplotlib
scikit-image              # SSIM cross-check in tests only (ADR-010)
pytesseract               # OCR evaluation (REQ-27)
python-Levenshtein        # CER
pycocotools               # keypoint annotation parsing (REQ-10)
kornia                    # BONUS ONLY — differentiable warp, named in spec §7
tqdm
```

**Not permitted:** `albumentations`, `imgaug`, `segmentation-models-pytorch`, `timm`,
`pytorch-msssim`, `lpips`, or anything supplying pretrained weights.
