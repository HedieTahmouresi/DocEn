# Project Brief

## What we are building

A document scanner, in the CamScanner sense: point a phone at a page and get back a flat, clean,
readable image. Built as **two independently-trained convolutional networks**, plus an optional
third stage that chains them.

```
  raw phone photo                                     clean scan-like image
        │                                                      ▲
        ▼                                                      │
┌───────────────────┐    4 corners    ┌──────────┐   rectified  ┌──────────────────┐
│  CORNER DETECTOR  │ ──────────────► │ HOMOGRAPHY│ ──────────► │  ENHANCEMENT NET │
│   (Task 2)        │                 │  + WARP   │    crop     │    (Task 1)      │
└───────────────────┘                 └──────────┘              └──────────────────┘
        └────────────── chained only in the BONUS ──────────────────────┘
```

**Task 1 — Enhancement network.** An encoder-decoder with skip connections that maps a degraded,
already-rectified document image to a clean scan. Trained on synthetic pairs. Judged by PSNR/SSIM
on synthetic data and by OCR readability on real photos.

**Task 2 — Corner detection network.** Predicts the four page corners from the raw photo. Two
formulations must both be built and compared: **(A)** direct coordinate regression through fully
connected layers, and **(B)** heatmap regression through an encoder-decoder. Judged by mean corner
localisation error and a strict all-four-corners success rate.

**Bonus.** Chain them into a fully automatic photo-to-scan pipeline, and measure how much corner
prediction error costs the enhancement stage.

## The central trick

There is no annotated training set, and building one is explicitly forbidden. Instead:

> Take a clean scan. Pick four random points on a random background photo. Warp the scan onto the
> background through that homography. Degrade the result until it looks like a phone photo.
>
> **The four points you chose are the corner labels.** And because the homography is known, warping
> the degraded composite *back* gives a pixel-aligned (degraded input, clean target) pair for the
> enhancement network.

One generator produces the training data and the labels for both tasks, with zero annotation
effort. This is `[REQ-07]` / `[REQ-08]`, and it is the intellectual core of the assignment.

## The central risk

Because all training data is synthetic, **the models are only as good as the degradation pipeline
is realistic.** This is not an abstract worry. The baseline implementation analysed in the research
report scored **96% corner success on synthetic test data and 0.00% on real photos**, with mean
error exploding from 1.85 px to 107.44 px. It had learned the generator, not the concept of a
document.

That failure — and the strategy for avoiding it — is documented in
`02-research/baseline-failure-analysis.md` and `02-research/sim2real-playbook.md`. **Read both
before writing the generator.** If you optimise one thing in this project, optimise for the
synthetic-to-real transfer, not for the synthetic leaderboard.

The stakes are structural: `[REQ-49]` — the final grade is assigned by running the pipeline on
**new, unseen realistic photos supplied by the teaching staff at the presentation.** Your own
real photos are a rehearsal for that, not the exam.

## What "done" looks like

| # | Deliverable | Requirement |
|---|---|---|
| 1 | Synthetic generator: OpenCV-only, six degradations, on-the-fly, verified aligned | REQ-07/33/34/35/37 |
| 2 | 10–15 real photos, corner-annotated in RoboFlow, each with a commercial reference scan | REQ-02/03/05 |
| 3 | Enhancement network, from scratch, encoder-decoder + skips, no dropout | REQ-19 |
| 4 | Loss-function comparison (MSE vs L1 vs L1+MS-SSIM vs +Sobel) | REQ-23/45 |
| 5 | PSNR/SSIM table: train / val / test + no-model baseline | REQ-25/26/47 |
| 6 | Real-photo evaluation: qualitative triplets + OCR readability vs commercial app | REQ-27/47 |
| 7 | Corner detector **Approach A** and **Approach B**, both trained, both measured | REQ-30/31 |
| 8 | Corner comparison: error, success rate, robustness, failure visualisations | REQ-31/45 |
| 9 | Two inference pipelines (rectified-image-in, raw-photo-in) | REQ-29/32/46 |
| 10 | Dropout ablation on all models; does the synthetic→real gap shrink? | REQ-38/39 |
| 11 | **Bonus:** chained end-to-end scanner, evaluated with annotated *and* predicted corners | REQ-40/41 |
| 12 | Report: plots, comparisons, limitations discussion | REQ-22/44/45/48 |
| 13 | *Optional bonus:* differentiable joint fine-tuning | REQ-42 |

Full grading mapping: `deliverables-checklist.md`.

## Scope boundaries — what this project is *not*

Do not build: diffusion models, GANs, transformers, attention-gated skips, perceptual/VGG losses
(also banned by `[CON-02]`), multi-document detection, curled-page dewarping, text detection or
recognition heads, self-supervised pretraining, or neural architecture search.

Curled and folded pages are explicitly out of scope — they belong in the *limitations* discussion
that `[REQ-48]` asks for, not in the model.

When tempted, load `05-skills/scope-guard.md`.

## Environment

Three machines, by necessity:

| Machine | Role | Notes |
|---|---|---|
| **This Linux box** | Home base | 4 cores, 7 GB RAM, **no GPU**, torch CPU-only. All non-training work: generator development, verification, annotation parsing, evaluation scripts, OCR, plotting, report. |
| **MX330 laptop** | Smoke tests + fallback | 4 GB VRAM, ~1.1 TFLOPS, Pascal, **no tensor cores** (AMP gives nothing). Roughly 15–20× slower than a T4. Use for "does it train at all" runs and as a no-timeout backup. |
| **Google Colab (T4)** | Primary training | ~8 TFLOPS FP32 + usable FP16 tensor cores. Weakness is the opposite: ~2 vCPUs, which will starve the GPU during on-the-fly generation. Session timeouts require checkpoint/resume. |

Consequences are binding and recorded in ADR-001: the repo must be portable (config-driven device
and batch size, no absolute paths), checkpoint every epoch, and resume cleanly. The CPU bottleneck
drives the data-pipeline design in ADR-003.

See `05-skills/portable-training.md`.

## Key agreed decisions (summary — authority is `01-decisions/`)

| | Decision | ADR |
|---|---|---|
| Framework | PyTorch | ADR-001 |
| Resolution | **512×512 for all three networks** | ADR-002 |
| Data generation | On-the-fly training + frozen on-disk val/test + cached decoded assets | ADR-003 |
| Backgrounds | ~50 self-shot phone photos (incl. cluttered hard negatives) + DTD | ADR-004 |
| Enhancement loss | Ablate MSE / L1 / L1+MS-SSIM / +Sobel; expect L1+MS-SSIM to win | ADR-006 |
| Corner detection | Both approaches, fairly trained, honest comparison | ADR-007 |
| SSIM | Implemented by hand, validated numerically against skimage | ADR-010 |
| Bonus | Tier 1 (chained) committed; Tier 2 (joint fine-tune) optional | ADR-012 |

## Source documents and their authority

| Document | Status |
|---|---|
| `Document Scanning Enhancement.md` | **Authoritative.** The graded specification. |
| `Document Scanner Implementation Plan.md` | **Advisory.** Useful analysis; contains at least one direct contradiction of the spec (it says to abandon Approach A; the spec mandates it — see `[REQ-30]`). Never cite it as a requirement. |
