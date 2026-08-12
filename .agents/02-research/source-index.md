# Source Index

Every external source consulted, what was taken from it, and how much weight it carries.

---

## Primary — retrieved and read

### Zhao, Gallo, Frosio, Kautz (2017) — *Loss Functions for Image Restoration with Neural Networks*
IEEE Transactions on Computational Imaging 3(1):47–57 · arXiv:1511.08861
<https://arxiv.org/abs/1511.08861> · <https://arxiv.org/pdf/1511.08861>

**Taken:** the mixed **MS-SSIM + L1** loss and the α = 0.84 weighting; the finding that the mixture
beats either component alone on both distortion and perceptual metrics.
**Caveat:** the paper's mixed loss applies a Gaussian weighting to the L1 term
(`L_mix = α·L_MS-SSIM + (1−α)·G_σM·L1`). Most reimplementations drop it. **Read §5 before
implementing** and state which form you used. → `loss-functions.md`, ADR-006.

### *Data Efficient Training of a U-Net Based Architecture for Structured Documents Localization* (SDL-Net)
arXiv:2310.00937 · <https://ar5iv.labs.arxiv.org/html/2310.00937>

Closest published match to this project's Approach B.
**Taken:** 512×512 input with **full-resolution 512×512, 4-channel** heatmaps; MSE loss; peak
extraction for coordinates; **max activation as a confidence score**; **quadrilateral IoU** as the
primary metric.
**Not applicable:** its MobileNetV2 encoder is ImageNet-pretrained — forbidden by `[CON-02]`.
→ `corner-localization.md`, ADR-008, ADR-011.

### Wang, Bovik, Sheikh, Simoncelli (2004) — *Image Quality Assessment: From Error Visibility to Structural Similarity*
Original SSIM. **Taken:** the definition, the 11×11 Gaussian window with σ=1.5, and C1/C2.
→ ADR-010.

### Wang, Simoncelli, Bovik (2003) — *Multiscale Structural Similarity for Image Quality Assessment*
**Taken:** MS-SSIM, the 5-scale structure, and the scale weights
`[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`. → ADR-010, ADR-006.

### Wang, Bo, Fuxin (2019) — *Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression*
ICCV 2019 · <https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.pdf>

**Taken (as context, not adopted):** the argument that MSE is a poor heatmap loss because positive
pixels number in the hundreds against tens of thousands of negatives; reported gains (81.8% → 85.9%
mean PCK@0.2 on LSP) and faster convergence.
**Status:** informs the foreground/background imbalance analysis in ADR-008. Not adopted by default
— escalate before using. → `corner-localization.md`.

### Luo et al. (2021) — *Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation*
arXiv:2012.15175 · **Taken:** the case that fixed σ is a simplification, and scale-adaptive σ as the
refinement. Supports treating σ as a tunable (`[ASM-05]`), not a constant.

### Describable Textures Dataset (DTD)
<https://www.robots.ox.ac.uk/~vgg/data/dtd/> · also on Kaggle and Hugging Face
5,640 images, 47 categories, 120 per category, 300×300 to 640×640. Released for research use.
**Taken:** the breadth half of the background pool. **Limitation noted:** flat close-up textures —
no clutter, no depth, no competing document edges — which is why ADR-004 pairs it with ~50
self-shot photos.

### MS-SSIM minimum size constraint
<https://github.com/francois-rozet/piqa/discussions/11> · also documented in `pytorch-msssim`,
`piq` and `torchmetrics`.
**Taken:** with a default 11×11 window and 5 scales, images must be ≥ `(11−1)·2⁴+1 = 161` px.
512 is comfortable. → ADR-010, `loss-functions.md`.

---

## Secondary — consulted, weaker weight

### Tesseract confidence miscalibration on enhanced images
Surfaced in OCR benchmarking literature on enhancement pipelines.
**Taken:** confidence scores are calibrated against Tesseract's training image statistics, so
CNN-enhanced images can score **lower confidence despite lower CER**.
**Consequence:** ADR-011 leads with CER and treats confidence as secondary, with the caveat stated
in the report. → `evaluation-and-ocr.md`.

### Tesseract image size / resolution guidance
GitHub issue tesseract-ocr/tesseract#3184 and the project's own documentation. ~300 DPI, ~10 px
x-height as practical minimums. Combined with A4 geometry this yields the ~2.6 px x-height at 512
figure — **that arithmetic is ours (Derived); check it if it matters.**

### CER/WER for OCR evaluation
Standard OCR-quality practice: `CER = Levenshtein / reference_length`; a CER of 10% means roughly
one character in ten is wrong, counting letters, punctuation and spaces. → ADR-011.

### Domain randomisation literature
Multiple sources on synthetic-to-real transfer (Tobin et al. and successors; recent surveys on
domain randomisation for detection).
**Taken:** randomising textures, lighting, viewpoints and post-processing makes a model treat real
data as one more variant; increased synthetic diversity — especially varied viewpoints and complex
backgrounds — is what bridges the gap. **This is the basis for widening ranges beyond observed
reality** in ADR-004 §3. → `sim2real-playbook.md`.

### SmartDoc 2015 (ICDAR) and document-localisation surveys
Context on the field: regression-based vs heatmap-based localisation, coarse-to-fine corner
refinement, multi-document corner association. **Refinement deliberately not adopted** — `[CON-10]`.

### `DocsaidLab/DocAligner`
<https://github.com/DocsaidLab/DocAligner> · An open-source four-corner document predictor. Useful
as a reference point for production practice. Not used as a source of implementation.

---

## Cited by the research report, **not independently verified**

Listed for traceability. Do not build decisions on these without checking them first.

- **`cv_project_notebook (78).ipynb`** — the baseline implementation. **We do not have this file.**
  Source of the 8.00% / 96.00% / 0.00% figures, 10.41 / 1.85 / 107.44 px errors, and the
  `margin_ratio=0.15`, `perspective_strength=0.08` parameters. All **Reported**, not Verified.
- **DocDiff** — *Document Enhancement via Residual Diffusion Models*. Out of scope (`[CON-10]`),
  noted as a state-of-the-art reference point.
- **DocNLC** — document enhancement with normalised and latent contrastive representation. Out of
  scope.
- **USCT-UNet** — on the "semantic gap" in U-Net skip connections. Real phenomenon; addressing it
  here would be over-engineering (ADR-005).
- Various MDPI/ResearchGate reviews of document image enhancement and binarisation — general
  background, nothing load-bearing.

---

## Named in the course spec itself

- RoboFlow keypoint annotation — <https://docs.roboflow.com/annotate/annotation-tools/keypoint-annotation>
- Kornia geometry transforms — <https://kornia.readthedocs.io/en/latest/geometry.transform.html>
- Tesseract OCR — <https://github.com/tesseract-ocr/tesseract>

These carry the spec's authority for *tooling choices*: using them is explicitly sanctioned.
