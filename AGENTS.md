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
├── GEMINI.md                 ← Operating contract (symlink to .agents/GEMINI.md)
├── README.md                 ← Human-facing project overview
├── requirements.txt          ← Python dependencies
│
├── configs/                  ← Configuration (YAML-based, layered)
│   ├── base.yaml             ← Single source of truth for ALL parameters
│   └── real_profile.yaml     ← Measured real-photo statistics for calibration
│
├── data/                     ← All datasets (not git-tracked)
│   ├── clean_scans/          ← 50 source document scans (ground truth)
│   ├── backgrounds/          ← 64 background textures for compositing
│   ├── real_photos/          ← Evaluation-only real smartphone photos
│   │   ├── raw/              ← 30 raw phone photos (active)
│   │   ├── reference_scans/  ← 30 CamScanner reference scans
│   │   └── annotations/      ← COCO polygon segmentation JSON
│   ├── splits/               ← Train/val/test scan assignments (splits.json)
│   └── frozen/               ← Pre-generated val/test sets (Phase 03)
│
├── src/                      ← All implementation code
│   ├── data/
│   │   ├── generator.py      ← Synthetic sample generator (OpenCV + NumPy only)
│   │   ├── datasets.py       ← PyTorch Dataset classes (RealPhotoDataset; more in Phase 03)
│   │   ├── splits.py         ← Hash-based split assignment
│   │   ├── annotations.py    ← COCO polygon annotation parsing, corner sorting
│   │   └── freeze.py         ← Frozen val/test writer (Phase 03)
│   ├── geometry/
│   │   └── homography.py     ← Corner ordering, quad validation, perspective warping
│   └── utils/
│       ├── config.py         ← YAML config loading and env profile merging
│       ├── seeding.py        ← Global + per-worker RNG utilities
│       ├── viz.py            ← Corner overlay visualization helpers
│       └── io.py             ← Image I/O with BGR/RGB boundary handling
│
├── tests/                    ← Automated test suite
│   ├── test_generator.py     ← Generator QA: round-trip, ordering, constraints
│   ├── test_homography.py    ← Homography and quad validation tests
│   └── test_real_photo_dataset.py ← RealPhotoDataset + CON-06 isolation test
│
├── scripts/                  ← Standalone utility scripts
│   ├── verify_generator.py   ← Phase 02 verification: samples, stranger test, histograms
│   ├── verify_real_annotations.py ← Phase 01 annotation verification
│   ├── compute_calibration_profile.py ← Real photo statistics → configs/real_profile.yaml
│   └── benchmark_cpu.py      ← CPU throughput benchmarking
│
├── outputs/                  ← Generated outputs (not git-tracked)
│   ├── figures/              ← Verification figures (p01_*, p02_*)
│   ├── checkpoints/          ← Saved model weights (Phase 04+)
│   └── logs/                 ← Training logs (Phase 04+)
│
├── notebooks/                ← Jupyter notebooks
└── .agents/                  ← Agent environment (planning, specs, state)
    ├── 00-project/           ← Requirements, constraints, quick-reference
    ├── 01-decisions/         ← Architecture Decision Records (ADR-001–016)
    ├── 02-research/          ← Research notes, baselines, sim2real strategy
    ├── 03-spec/              ← Detailed specs (generator, datasets, models, etc.)
    ├── 04-phases/            ← Phase definitions and gate criteria
    ├── 05-skills/            ← Agent skill files (QA, diagnostics, etc.)
    ├── 06-workflow/          ← Git workflow, escalation protocol
    └── state/                ← STATUS.md, session log, experiments, discoveries
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

**Phase:** Phase 05 (Enhancement Real Photo Evaluation & OCR) & Phase 07 (Dropout Ablation)

**Completed:**
- ✅ Repository structure and config.py
- ✅ M1: Environment and data setup (clean scans, backgrounds collected)
- ✅ M2: Synthetic data pipeline (degradations, transforms, generator)
- ✅ M3: Real photo collection, annotation, and reference scans (30 raw photos + reference scans)
- ✅ M4: Dataset classes, splits, frozen sets, DataLoader pipeline
- ✅ Phase 04: Enhancement loss ablation (`exp-005..008` trained, `exp-008` SSIM 0.8497 PASSED)
- ✅ Phase 06: Corner detection paired training & evaluation (`exp-009` & `exp-010`, `exp-010` Val MCE 1.05 px, 99.8% Succ@1% PASSED)
- ✅ 16 Architecture Decision Records (ADR-001 through ADR-016)

**In Progress / Next:**
- ⬜ Phase 05: Enhancement evaluation on real smartphone photos & OCR metrics (PSNR, SSIM, CER)
- ⬜ Phase 07: Dropout regularization ablation comparison

> For detailed status, see `.agents/state/STATUS.md`.
> For experiment logs, see `.agents/state/experiments.md`.


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
