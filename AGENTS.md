# AGENTS.md — Engineering Entry Point

> **Every agent reads this file first. No exceptions.**
>
> This is the single authoritative entry point for all contributors — human or LLM.
> It tells you what this project is, how it's organized, what rules you must follow,
> what the current state is, and what to do when you're done.

---

## 1. Project Identity

This project builds the core engine of a **CamScanner-like document scanning application** for a university Computer Vision course. The engine has two independently trained CNN components:

1. **Corner Detection Network** — detects the 4 page corners from a raw phone photo
2. **Enhancement Network** — transforms a degraded, perspective-rectified document crop into a clean scan

All training data is **generated synthetically** by warping clean document scans onto random backgrounds with realistic degradations. The models are evaluated on **real smartphone photos** and graded by teaching staff on **unseen photos at presentation**.

```
Raw Phone Photo → Corner Detector → Perspective Rectification → Enhancement Network → Clean Scan
```

> **Critical:** The two networks are trained and evaluated independently. The enhancement
> network only sees rectified crops, never raw photos. They only chain together at inference
> in the bonus end-to-end pipeline.

---

## 2. Repository Map

```
Project/
├── AGENTS.md                 ← YOU ARE HERE
├── README.md                 ← Human-facing project overview
├── config.py                 ← Single source of truth for ALL parameters
├── requirements.txt          ← Python dependencies
│
├── docs/                     ← Engineering documentation (local workspace)
│   ├── ARCHITECTURE.md       ← Technical vision, pipeline design, data philosophy
│   ├── CONVENTIONS.md        ← Coding rules, naming, imports, git workflow
│   ├── DECISIONS.md          ← Architecture Decision Records (ADR-001 through ADR-016)
│   ├── STATUS.md             ← Current project state and implementation progress
│   └── ROADMAP.md            ← Milestones, dependencies, and phase planning
│
├── data/                     ← All datasets (not git-tracked)
│   ├── clean_scans/          ← 50 source document scans (ground truth)
│   ├── backgrounds/          ← 47 background textures for compositing
│   ├── real_photos/          ← Evaluation-only real smartphone photos
│   │   ├── raw/              ← 35 raw phone photos
│   │   ├── reference_scans/  ← 35 CamScanner reference scans
│   │   └── annotations/      ← COCO polygon segmentation JSON
│   ├── splits/               ← Train/val/test scan assignments
│   └── frozen/               ← Pre-generated val/test sets
│
├── data_generation/          ← Synthetic data pipeline
│   ├── transforms.py         ← Perspective warp, homography, inverse warp
│   ├── degradations.py       ← All degradation functions (OpenCV only)
│   └── generator.py          ← SyntheticSampleGenerator class
│
├── datasets/                 ← PyTorch Dataset classes
│   ├── synthetic_dataset.py  ← On-the-fly and frozen dataset classes
│   └── real_photo_dataset.py ← Real photo evaluation datasets
│
├── models/                   ← Neural network architectures
│   ├── enhancement_net.py    ← Task 1: Encoder-decoder with skip connections
│   ├── corner_regression.py  ← Task 2A: CNN → FC → 8 coordinates
│   └── corner_heatmap.py     ← Task 2B: Encoder-decoder → 4 heatmaps
│
├── training/                 ← Training infrastructure
│   ├── losses.py             ← L1 + SSIM + Edge composite loss
│   ├── train_enhancement.py  ← Enhancement network training loop
│   └── train_corner.py       ← Corner detection training loop
│
├── evaluation/               ← Evaluation infrastructure
│   ├── metrics.py            ← PSNR, SSIM, corner error, success rate
│   ├── evaluate_enhancement.py
│   ├── evaluate_corner.py
│   └── ocr_eval.py           ← Tesseract-based readability evaluation
│
├── pipelines/                ← Inference pipelines
│   ├── enhancement_pipeline.py
│   ├── corner_pipeline.py
│   └── end_to_end.py         ← Bonus: full photo-to-scan pipeline
│
├── utils/                    ← Shared utilities
│   ├── io_utils.py           ← Image I/O, annotation parsing, corner sorting
│   └── visualization.py      ← Plotting and visualization helpers
│
├── scripts/                  ← Standalone utility scripts
│   ├── freeze_splits.py      ← Generate frozen val/test sets
│   ├── verify_pipeline.py    ← Verify synthetic data pipeline
│   └── visualize_samples.py  ← Visualize dataset samples
│
├── outputs/                  ← Generated outputs (not git-tracked)
│   ├── checkpoints/          ← Saved model weights
│   ├── logs/                 ← Training logs
│   └── visualizations/       ← Verification images
│
├── notebooks/                ← Jupyter notebooks
└── .agents/                  ← Agent workspace (not git-tracked)
    ├── reports/              ← Agent completion reports
    └── report-template.md    ← Report template
```

---

## 3. Non-Negotiable Rules

These constraints are **architectural decisions** that must not be changed without explicit approval from the lead engineer. They exist because the project specification requires them or because they prevent known failure modes.

### Data & Pipeline
- **Config over constants.** ALL parameters come from `config.py`. Import as `from config import CFG`. Never hardcode magic numbers.
- **OpenCV only for degradations.** No Albumentations, imgaug, or torchvision transforms in the degradation pipeline.
- **No image flipping.** Never flip horizontally or vertically. Mirrored text is not a valid degradation.
- **Split by source scan.** Two degraded versions of the same page must never cross split boundaries.
- **Frozen val/test sets.** Validation and test sets are generated once with a fixed seed and saved to disk.
- **Real photos are evaluation-only.** Never train on them. Never run the degradation pipeline on them.

### Models & Training
- **PyTorch.** All models, datasets, and training loops use PyTorch.
- **No pre-trained weights.** All models are trained from scratch.
- **No imported architectures.** Build from `nn.Conv2d`, `nn.MaxPool2d`, etc. No `import UNet`.
- **No dropout in initial versions.** Dropout is added only in the regularization phase (Section 6).
- **Device-agnostic code.** Always use `CFG.DEVICE` and `.to(device)`. Never assume CUDA.

### Corner Ordering
- **TL → TR → BR → BL (clockwise).** This is the universal corner convention.
- **Annotations are COCO polygon segmentation**, not keypoints. Polygon vertices arrive in arbitrary order and must be sorted programmatically using the centroid-based algorithm in `utils/io_utils.py`.
- A single ordering mistake silently flips or rotates the rectified output.

### Image Format
- OpenCV loads BGR; PyTorch expects RGB. Convert at the boundary (inside Dataset classes).
- All internal representations: RGB float32 in [0, 1].
- No ImageNet mean/std normalization. Input and output are both in [0, 1].
- Corner coordinates normalized to [0, 1] by dividing by image dimensions.

---

## 4. Current Project State

**Phase:** Data Pipeline Implementation (Milestones M1, M2, M4)

**Completed:**
- ✅ Repository structure and config.py
- ✅ M1: Environment and data setup (clean scans, backgrounds collected)
- ✅ M3: Real photo collection, annotation, and reference scans (35 photos)
- ✅ 16 Architecture Decision Records

**In Progress / Next:**
- ⬜ M2: Synthetic data pipeline (degradations, transforms, generator)
- ⬜ M4: Dataset classes, splits, frozen sets, DataLoader pipeline

**All Python modules** (except `config.py` and `utils/io_utils.py` docstrings) **contain only placeholder docstrings — zero implementation exists yet.**

> For detailed status, see `docs/STATUS.md`.
> For the full milestone roadmap, see `docs/ROADMAP.md`.

---

## 5. Reading Order for New Agents

1. **This file** (AGENTS.md) — you're reading it
2. **`docs/CONVENTIONS.md`** — coding rules, naming, git workflow
3. **`config.py`** — understand every parameter
4. **`docs/ROADMAP.md`** — find your assigned milestone
5. **Module docstrings** — read the target file's docstring before implementing
6. **`docs/ARCHITECTURE.md`** — if you need deeper technical context
7. **`docs/DECISIONS.md`** — if you need to understand why a decision was made

---

## 6. Session Protocol

### Starting a Session
1. Read this file (AGENTS.md)
2. Read `docs/CONVENTIONS.md`
3. Check `docs/STATUS.md` for current state
4. Identify your task from `docs/ROADMAP.md`

### During Implementation
- Import all parameters from `config.py`
- Read the target module's docstring before writing code
- If you face a design decision not covered in `docs/DECISIONS.md`, flag it — do not decide unilaterally
- Commit after each completed unit of work using Conventional Commits
- Verify your work (run it, visualize outputs, check shapes/ranges)

### Ending a Session
Write a completion report in `.agents/reports/` using the template in `.agents/report-template.md`. Name it `YYYY-MM-DD_<milestone>_<description>.md`.

The report captures:
- What was implemented
- What was verified and how
- Any issues encountered
- Design decisions made (if any)
- What the next agent should pick up

> These reports are the project's institutional memory. Future agents read them to understand what happened and why.

---

## 7. Boundaries

**What implementation agents are free to change:**
- Implementation details within modules (algorithms, data structures, optimizations)
- Parameter values in `config.py` (with documented rationale)
- Adding new utility functions
- Adding new visualization/verification scripts

**What requires lead engineer approval:**
- Adding or removing modules/files from the repository structure
- Changing the pipeline architecture (how components connect)
- Changing any rule in Section 3 of this document
- Adding new dependencies to `requirements.txt`
- Modifying the corner ordering convention
- Changing the data split strategy
- Using pre-trained weights or imported architectures
