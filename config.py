"""
Central Configuration for Document Scanning & Enhancement Project
=================================================================

ALL project parameters are defined here. No magic numbers anywhere else.
Every module imports from this file. If you need to change a parameter,
change it HERE and only here.

Usage:
    from config import CFG
    print(CFG.IMG_SIZE)  # 256

Convention:
    - All paths are pathlib.Path objects
    - All ranges are tuples (min, max)
    - Parameters are grouped by category
    - Comments explain WHY a value was chosen, not just what it is
"""

import torch
from pathlib import Path


class CFG:
    """Project-wide configuration. Import as: from config import CFG"""

    # ========================================================================
    # PROJECT PATHS
    # ========================================================================
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    CLEAN_SCANS_DIR = DATA_DIR / "clean_scans"
    BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
    REAL_PHOTOS_DIR = DATA_DIR / "real_photos"
    RAW_PHOTOS_DIR = REAL_PHOTOS_DIR / "raw"
    REFERENCE_SCANS_DIR = REAL_PHOTOS_DIR / "reference_scans"
    ANNOTATIONS_DIR = REAL_PHOTOS_DIR / "annotations"
    # Annotation format: COCO polygon segmentation (NOT keypoints)
    # The segmentation field contains [x1,y1, x2,y2, x3,y3, x4,y4, x1,y1]
    # Vertices are NOT in TL→TR→BR→BL order — must be sorted programmatically
    # The RoboFlow COCO export lives in a subdirectory structure
    ANNOTATIONS_FILE = ANNOTATIONS_DIR / "CV Doc-Enhancement Real Test Set.v1i.coco" / "train" / "_annotations.coco.json"
    SPLITS_DIR = DATA_DIR / "splits"
    FROZEN_DIR = DATA_DIR / "frozen"
    FROZEN_VAL_DIR = FROZEN_DIR / "val"
    FROZEN_TEST_DIR = FROZEN_DIR / "test"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
    LOGS_DIR = OUTPUT_DIR / "logs"
    VIS_DIR = OUTPUT_DIR / "visualizations"

    # ========================================================================
    # IMAGE DIMENSIONS
    # ========================================================================
    # 512x512 resolution for higher quality training.
    # Note: Using larger image sizes significantly increases VRAM usage.
    IMG_SIZE = 512
    IMG_CHANNELS = 3  # RGB

    # ========================================================================
    # CORNER CONVENTION
    # ========================================================================
    # CRITICAL: Corners MUST be ordered consistently everywhere.
    # Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left (clockwise)
    # This matches OpenCV convention and the project spec.
    # Coordinates are always normalized to [0, 1] after preprocessing.
    NUM_CORNERS = 4
    CORNER_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
    # Shape of corner tensor: (4, 2) -> 4 corners, each (x, y) normalized

    # ========================================================================
    # DATA SPLITS
    # ========================================================================
    # Split by SOURCE SCAN, not by generated sample.
    # Two degraded versions of the same page must NEVER cross splits.
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    SPLIT_SEED = 42  # Seed for reproducible train/val/test assignment

    # ========================================================================
    # FROZEN VALIDATION/TEST SETS
    # ========================================================================
    # Val and test sets are generated ONCE with a fixed seed and saved to disk.
    # This ensures every epoch and every model is scored on identical images.
    FROZEN_SEED = 123  # Separate from split seed for independence
    FROZEN_VAL_SAMPLES_PER_SCAN = 5   # Total val samples = val_scans * 5
    FROZEN_TEST_SAMPLES_PER_SCAN = 5  # Total test samples = test_scans * 5

    # ========================================================================
    # DEGRADATION PIPELINE
    # All degradation parameters are tunable ranges [min, max].
    # Every sample randomizes within these ranges.
    # ========================================================================

    # --- Step 1: Perspective Warp ---
    # Corner displacement as fraction of image dimension.
    # 0.05 = subtle tilt, 0.25 = aggressive perspective.
    # Too extreme creates thin slivers; too mild doesn't teach perspective.
    WARP_CORNER_RANGE = (0.05, 0.25)

    # --- Step 2: Resolution Loss (Downscale-Upscale) ---
    # Simulates photographing from far away.
    # Factor of 2 = mild blur, 4 = heavy information loss.
    DOWNSCALE_RANGE = (2.0, 4.0)
    # Interpolation for downscale: INTER_AREA (best for shrinking)
    # Interpolation for upscale: INTER_LINEAR (simulates real upscaling)

    # --- Step 3: Brightness, Contrast, Color Cast ---
    BRIGHTNESS_RANGE = (0.5, 1.5)    # Multiplicative factor
    CONTRAST_RANGE = (0.5, 1.5)      # Multiplicative factor around mean
    COLOR_CAST_R_RANGE = (0.85, 1.15) # Scale red channel
    COLOR_CAST_B_RANGE = (0.85, 1.15) # Scale blue channel
    # Green channel stays at 1.0 (anchored) to avoid overall brightness shift

    # --- Step 4: Illumination Gradients & Shadows ---
    # Gradient: smooth brightness variation across the image
    GRADIENT_INTENSITY_RANGE = (0.3, 1.0)  # Min brightness at darkest point
    # Shadows: blurred polygons overlaid at reduced intensity
    SHADOW_COUNT_RANGE = (1, 3)       # Number of shadow shapes per image
    SHADOW_INTENSITY_RANGE = (0.3, 0.7)  # Shadow darkness (0=black, 1=transparent)
    SHADOW_BLUR_KERNEL_RANGE = (51, 151)  # Blur kernel size (odd numbers only)
    SHADOW_NUM_VERTICES_RANGE = (3, 7)    # Polygon vertex count

    # --- Step 5: Blur & Noise ---
    BLUR_KERNEL_SIZES = [3, 5, 7]     # Randomly chosen per sample
    BLUR_SIGMA_RANGE = (0.5, 2.0)     # Gaussian blur sigma
    NOISE_SIGMA_RANGE = (5.0, 25.0)   # Gaussian noise std (on 0-255 scale)

    # --- Step 6: JPEG Compression ---
    JPEG_QUALITY_RANGE = (30, 80)     # cv2.imencode quality parameter

    # ========================================================================
    # TRAINING — ENHANCEMENT NETWORK (Task 1)
    # ========================================================================
    ENHANCE_BATCH_SIZE = 4  # Reduced to fit 512x512 in 4GB VRAM
    ENHANCE_LR = 1e-3
    ENHANCE_EPOCHS = 100
    ENHANCE_SAMPLES_PER_EPOCH = 2000  # Effective training set size per epoch
    ENHANCE_OPTIMIZER = "adam"         # Options: "adam", "adamw"
    ENHANCE_SCHEDULER = "cosine"      # Options: "cosine", "step", "none"
    ENHANCE_SCHEDULER_T_MAX = 100     # For cosine annealing

    # ========================================================================
    # TRAINING — CORNER DETECTION (Task 2)
    # ========================================================================
    CORNER_BATCH_SIZE = 4  # Reduced to fit 512x512 in 4GB VRAM
    CORNER_LR = 1e-3
    CORNER_EPOCHS = 100
    CORNER_SAMPLES_PER_EPOCH = 2000
    CORNER_OPTIMIZER = "adam"
    CORNER_SCHEDULER = "cosine"
    CORNER_SCHEDULER_T_MAX = 100
    # Heatmap target: Gaussian sigma in pixels (at IMG_SIZE resolution)
    # 5 pixels is narrow enough for precise localization but wide enough
    # to provide gradient signal during training.
    CORNER_HEATMAP_SIGMA = 5.0

    # ========================================================================
    # LOSS FUNCTION WEIGHTS (Enhancement Network)
    # ========================================================================
    # Composite loss = w1*L1 + w2*(1-SSIM) + w3*EdgeL1
    # L1: primary reconstruction loss, sharper than MSE
    # SSIM: perceptual quality, preserves structure
    # Edge: L1 on Sobel edge maps, preserves text strokes
    LOSS_L1_WEIGHT = 1.0
    LOSS_SSIM_WEIGHT = 0.5
    LOSS_EDGE_WEIGHT = 0.1

    # ========================================================================
    # NORMALIZATION
    # ========================================================================
    # Images: divide by 255 to [0, 1]. No ImageNet mean/std normalization
    # because the output must also be in [0, 1] (symmetric input/output).
    # Corners: divide by image dimensions to [0, 1] (resolution-independent).

    # ========================================================================
    # DROPOUT (Section 6 — applied AFTER initial training)
    # ========================================================================
    DROPOUT_RATE = 0.3  # Default dropout probability
    # Where to apply: FC layers (regression), decoder (encoder-decoder)

    # ========================================================================
    # DEVICE & REPRODUCIBILITY
    # ========================================================================
    # Note: Local GPU is an MX330 with 4GB VRAM.
    # Code must run seamlessly on CPU, local MX330, and Colab (T4/V100).
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Adjust batch sizes based on available VRAM
    # MX330 (4GB): use batch_size 2-4 for 512x512 images
    # Colab T4 (15GB): use batch_size 16-32 
    # If CUDA OOM, reduce batch sizes here
    # Consider using a smaller batch size (e.g. 2 or 4) when running on the local MX330.
    MAX_LOCAL_BATCH_SIZE = 4
    GLOBAL_SEED = 42

    # ========================================================================
    # EVALUATION
    # ========================================================================
    # Corner detection success thresholds (in pixels at IMG_SIZE)
    CORNER_SUCCESS_THRESHOLDS = [5, 10, 20]  # pixels
    # OCR engine
    OCR_ENGINE = "tesseract"  # pytesseract

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        for attr_name in dir(cls):
            if attr_name.endswith('_DIR'):
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)
