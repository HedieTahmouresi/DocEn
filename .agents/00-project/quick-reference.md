# Quick Reference

Pure lookup. Every value here is decided elsewhere — the **Source** column is authoritative and this
page is a convenience. If they ever disagree, the source wins.

---

## Fixed numbers

| Quantity | Value | Source |
|---|---|---|
| Working resolution, **all three networks** | **512 × 512** | ADR-002 |
| Image diagonal @512 (for % -of-diagonal metrics) | 724.1 px | ADR-011 |
| Split | 80 / 10 / 10, **by source scan** | `[REQ-14]` |
| Frozen val / test target size | ~500 samples each | ADR-003 |
| U-Net levels / base channels | 4 / 64 | ADR-005 |
| Bottleneck spatial size | 32 × 32 | ADR-005 |
| Heatmap σ | **8 px @512** (sweep {4, 8, 12}) | ADR-008, `[ASM-05]` |
| Heatmap render window | **±3σ, pasted** — never full-frame | ADR-008 |
| Soft-argmax window | 11 × 11 around argmax | ADR-008 |
| MS-SSIM weight α | **0.84** (sweep {0.7, 0.84, 0.95}) | ADR-006, `[ASM-04]` |
| Sobel weight λ | 0.1 | ADR-006 |
| MS-SSIM scales | 5, weights `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]` | ADR-010 |
| MS-SSIM minimum image size | 161 px | ADR-010 |
| SSIM window | 11×11 Gaussian, σ=1.5, K1=0.01, K2=0.03 | ADR-010 |
| `data_range` for all metrics | **1.0** (metrics in `[0,1]`) | ADR-009 |
| Corner success thresholds | strict 1% diag (≈7.2 px) · lenient 2% (≈14.5 px) | ADR-011 |
| OCR evaluation resolution | long side ~2000 px, all images matched | ADR-011 |
| CER documents to transcribe | 5 | ADR-011 |
| Round-trip alignment gate | **PSNR > 30 dB** (photometrics off) | generator-spec §3 |
| GPU utilisation gate | **≥ ~50%** sustained on Colab | ADR-003 |
| Optimizer / LR | Adam, 1e-3, **`weight_decay=0.0`** | training-spec |
| Real photos to capture | 10–15 required; **20–25 recommended** | `[REQ-02]`, `[OPEN-02]` |
| Background photos | ~50, **≥15 cluttered** | ADR-004 |

## Degradation ranges (provisional — recalibrate in Phase 01)

| Parameter | Range | |
|---|---|---|
| Page area fraction | 0.15 – 0.95 | ← the baseline's fixed 0.15 margin was the failure |
| In-plane rotation | ±25° | |
| Perspective strength | 0.0 – 0.35 | ← the baseline's fixed 0.08 was the failure |
| Downscale factor | ×2 – ×4 | `[REQ-34]` |
| Contrast / brightness | 0.7–1.3 / ±40 | |
| Channel gain (cast) | 0.9 – 1.1, anti-correlated R/B | |
| Illumination gradient | 0.55 – 1.15 | |
| Shadows | 0–3, opacity 0.15–0.55, blur 15–91 | |
| Blur kernel / noise σ | 3–9 px / 2–12 | |
| JPEG quality | **30 – 80** | `[REQ-34]` |

After Phase 01: set ranges to cover the measured real-photo distribution, **then widen ~1.5–2×**
(ADR-004 §3).

---

## Corner order — memorise this

```
  0 ────────── 1     0 = TL   1 = TR   2 = BR   3 = BL
  │            │     Clockwise from top-left OF THE PAGE,
  │            │     not of the image.
  3 ────────── 2
```

Colours for every overlay: **TL red · TR green · BR blue · BL yellow.** A bowtie means wrong order.
Never sort predicted corners to "fix" ordering. — `conventions.md` §1, §8

---

## Tensor conventions

| Stage | Layout | dtype | Range | Colour |
|---|---|---|---|---|
| Disk / `cv2.imread` | HWC | uint8 | 0–255 | **BGR** |
| Degradation pipeline | HWC | uint8 | 0–255 | **BGR** |
| Model input | NCHW | float32 | **standardised** | RGB |
| Target / output / metrics | NCHW | float32 | **`[0,1]`** | RGB |
| `cv2.imwrite` | HWC | uint8 | 0–255 | **BGR** |

Points are `(x, y)` = `(col, row)`. NumPy indexes `img[y, x]`. **In BGR, index 2 is red.**
— `conventions.md` §2, §3

---

## The prohibitions

| ✗ | Detail |
|---|---|
| Pretrained weights | Anywhere. **Also rules out VGG/LPIPS perceptual losses** — `[CON-02]` |
| Pre-built architectures | No `smp`, `timm`, `torchvision.models`, even untrained — `[CON-01]` |
| Third-party transform libs | Degradation pipeline is **OpenCV + NumPy only** — `[CON-03]` |
| Dropout / weight decay | Zero in Phases 04 and 06. **`AdamW` defaults to 0.01** — `[CON-04]` |
| Flips | Mirrored text — `[CON-05]` |
| Training on real photos | Never. Never degrade them either — `[CON-06]` |
| Touching the test split | Until final evaluation — `[CON-07]` |
| Corner coords into enhancement | It sees the rectified crop only — `[CON-08]` |
| Reordering the degradations | The §4.3 sequence is fixed — `[CON-09]` |

---

## The traps that fail silently

| Trap | Where |
|---|---|
| Re-deriving the inverse homography from corners instead of inverting the matrix | generator-spec §3 |
| Worker RNG fork — all workers generate identical samples | conventions §5 |
| Corner ordering wrong → homography flips the page | conventions §1 |
| Batch-pooled PSNR instead of per-image | evaluation-spec §1 |
| `skimage` SSIM default is uniform 7×7, not Gaussian 11×11 | ADR-010 |
| MS-SSIM under AMP → NaN (cast to float32) | training-spec §3 |
| MS-SSIM sign — the loss is `1 − MS-SSIM` | ADR-006 |
| Frozen sets saved as JPEG (double compression) | ADR-003 |
| Mixing `frozen_version`s in one table | experiment-discipline |
| Aspect-ratio policy differing between train and inference | generator-spec §4 |
| EXIF rotation ignored by `cv2.imread` | data-contract §4 |
| OCR at each image's natural resolution | ADR-011 §5 |

---

## Where to look

| Question | File |
|---|---|
| What am I doing next? | `state/STATUS.md` |
| Is X required? | `00-project/requirements.md` — **if it's not there with a citation, it's not a requirement** |
| Am I allowed to do X? | `00-project/constraints.md` |
| Why is X the way it is? | `01-decisions/DECISIONS.md` → the ADR |
| What's the evidence for X? | `02-research/` |
| How do I build X? | `03-spec/` |
| Am I done? | the phase gate in `04-phases/` |
| How do I not screw this up? | `05-skills/` |
| I want to change something agreed | `06-workflow/escalation-protocol.md` |
| Am I finished overall? | `00-project/deliverables-checklist.md` |
