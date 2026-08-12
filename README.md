# Document Scanning & Enhancement

A CNN-based document scanning engine built from scratch for a university Computer Vision course. The engine detects document corners from phone photos, rectifies the perspective, and enhances the image to produce a clean scan — replicating the core functionality of apps like CamScanner.

## Architecture

```
Raw Phone Photo → Corner Detector → Perspective Rectification → Enhancement Network → Clean Scan
```

The pipeline has two independently trained CNN components:

1. **Corner Detection Network** — locates the 4 page corners in a raw photo
2. **Enhancement Network** — transforms a degraded, rectified document crop into a clean scan

Both networks are trained entirely on **synthetically generated data** — clean document scans warped onto random backgrounds with realistic degradations. They are evaluated on **real smartphone photos**.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify the environment
python -c "from config import CFG; print(f'Device: {CFG.DEVICE}')"
```

### Hardware

| Environment | GPU | Use For |
|-------------|-----|---------|
| Local | MX330 (2GB VRAM) | Development, debugging |
| Google Colab | T4 (15GB VRAM) | Training, evaluation |
| CPU | — | Fallback |

All code is device-agnostic via `CFG.DEVICE`.

## Project Structure

```
Project/
├── config.py                 ← Single source of truth for ALL parameters
├── AGENTS.md                 ← Entry point for AI/LLM contributors
├── data/                     ← Datasets (not git-tracked)
├── data_generation/          ← Synthetic data pipeline
├── datasets/                 ← PyTorch Dataset classes
├── models/                   ← Neural network architectures
├── training/                 ← Loss functions and training loops
├── evaluation/               ← Metrics and evaluation scripts
├── pipelines/                ← Inference pipelines
├── utils/                    ← I/O and visualization utilities
├── scripts/                  ← Standalone utility scripts
├── outputs/                  ← Checkpoints, logs, visualizations
└── notebooks/                ← Jupyter notebooks
```

## Key Constraints

- **PyTorch** — all models, datasets, and training
- **OpenCV only** — for degradation pipeline (no Albumentations, imgaug)
- **No pre-trained weights** — all models trained from scratch
- **No imported architectures** — built from `nn.Conv2d`, `nn.MaxPool2d`, etc.
- All images: RGB float32 in [0, 1]
- Corner convention: TL → TR → BR → BL (clockwise)
- Data splits by source scan (prevents leakage)

## For Contributors

- **AI agents:** Start with [AGENTS.md](AGENTS.md)
- **Design decisions:** See `docs/DECISIONS.md` (16 ADRs)
- **Configuration:** All parameters in [config.py](config.py)
