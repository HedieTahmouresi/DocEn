# Corner Localisation: Regression vs Heatmaps

Background for ADR-007 and ADR-008.

---

## The two formulations

`[REQ-30]` requires building both. This is the *why*.

### Approach A — direct coordinate regression

A convolutional encoder extracts features; the final feature map is flattened and passed through
fully connected layers producing 8 numbers.

**The structural objection**, and the thing to be able to state at the presentation: convolutional
layers are translation-**equivariant** — shift the input, the feature map shifts identically. A
fully connected layer applied to a flattened feature map destroys that. Each output weight is tied
to an absolute position in the flattened vector, so the network must learn a global, highly
non-linear mapping from "what the whole image looks like" to "the corner is at (0.31, 0.78)". That
mapping has to be re-learned, in effect, for every possible page position, and it is sensitive to
small shifts.

Localisation is intrinsically a spatial task, and this architecture converts it into a
non-spatial one.

**Secondary failure mode:** nothing constrains the *ordering* of the 8 outputs. Regression models
can emit corners in an inconsistent order, producing a homography that folds or mirrors the page.
Do not paper over this by sorting the outputs — sorting hides a real error and breaks on rotated
pages (`00-project/conventions.md` §1).

**Prior evidence:** the analysed baseline reached 8.00% success / 10.41 px mean error on synthetic
data — near-failure on the *easy* distribution.

### Approach B — heatmap regression

A fully convolutional encoder-decoder produces four single-channel maps, one per corner, each
trained to contain a Gaussian blob at the true location. Coordinates come out by peak extraction.

**Why it fits better:** the network stays fully convolutional, so spatial topology is preserved end
to end. The task becomes dense per-pixel prediction — "does a corner live here?" — which is local,
translation-equivariant, and exactly what convolution is good at. Output ordering is structural: a
corner's identity is its *channel index*, not a position in a vector.

**Costs:** a decoder (more compute and memory than an FC head), a non-differentiable argmax at
inference unless soft-argmax is used, quantisation limited by heatmap resolution, and a severe
foreground/background class imbalance (ADR-008).

**Prior evidence:** 96.00% success / 1.85 px on the same synthetic data as Approach A's 8%. That
gap is a genuine signal about representation — even though the 96% was itself inflated by a narrow
generator (`baseline-failure-analysis.md`).

---

## What the literature does

**SDL-Net** — *Data Efficient Training of a U-Net Based Architecture for Structured Documents
Localization* (arXiv:2310.00937). The closest published match to this project's Approach B:

- 512×512 input, 3 channels
- **512×512 output heatmaps, 4 channels** — full resolution, no output stride
- **MSE** loss on the heatmaps
- coordinates by extracting the **activation peak** per channel
- **max activation value used as a confidence score**
- U-Net structure with a MobileNetV2 encoder (**pretrained — not available to us, `[CON-02]`**)
- augmentation: crop, resize, rotation, perspective, global illumination scaling, Gaussian noise
- primary metric: **IoU of the predicted quadrilateral** vs ground truth
- notable data-efficiency result: pretraining on 4 document classes and fine-tuning on a 5th lifted
  IoU from 0.58 to 0.73

Directly adopted here: full-resolution 4-channel heatmaps, MSE as the starting loss, peak
extraction, confidence from peak value, and quad IoU as a supplementary metric (ADR-008, ADR-011).

**Broader keypoint literature.** Human pose estimation is where heatmap regression matured:
- Typical convention is an output stride of 4 (e.g. 64×64 heatmaps from 256×256 input) with σ ≈ 2
  px on the heatmap — equivalent to **σ ≈ 8 px in input space**, which is where ADR-008's σ=8 comes
  from.
- Output stride is known to matter a lot: heatmap methods are strongly affected by it because of
  quantisation error at extraction. Full resolution sidesteps this.
- **MSE is a known-poor choice for heatmap regression** because the loss is dominated by the vast
  background region. Adaptive Wing Loss (Wang et al., ICCV 2019) was designed for exactly this and
  reports substantial gains (e.g. 81.8% → 85.9% mean PCK@0.2 on LSP) plus faster convergence.
  ADR-008 keeps MSE as the default anyway — it matches the spec's phrasing and is proven to work on
  this task — with foreground-weighted MSE as a pre-approved fallback and Adaptive Wing behind an
  escalation.

**Document localisation specifically.** The SmartDoc 2015 (ICDAR) benchmark is the standard
evaluation set for document localisation in video/photos. Published approaches split into
regression-based (direct coordinate output) and heatmap-based (typically hourglass architectures),
with refinement variants that crop patches around coarse corner predictions and re-localise inside
each patch for sub-pixel accuracy.

**Corner refinement is deliberately not adopted here.** It is a real technique and it works, but it
adds four extra forward passes and a second model for accuracy this project does not need. `[CON-10]`
— note it in `[REQ-48]` as a potential improvement instead.

---

## Coordinate extraction, compared

| Method | Precision | Robustness to a second peak | Differentiable |
|---|---|---|---|
| `argmax` | integer only | high — ignores everything but the max | no |
| Global soft-argmax | sub-pixel | **low** — a spurious peak drags the estimate toward the midpoint | yes |
| **argmax + local soft-argmax** | sub-pixel | high | yes, w.r.t. heatmap values |
| Contour centroid | sub-pixel | medium | awkward |

ADR-008 chooses the third. The failure mode of global soft-argmax matters concretely here: this
project *deliberately* trains on cluttered backgrounds with competing quadrilaterals (ADR-004), so
secondary peaks are an expected output, not a rare pathology. Taking an expectation over the whole
map would place the corner in empty space between two candidates.

The research report's suggestion — "the mathematical centroid of the predicted heatmap contour" —
is the same instinct; local soft-argmax is the cleaner formulation.

---

## Making the comparison fair

The expected result (B wins) is so widely assumed that it is easy to produce it by accident through
a weak Approach A. ADR-007 §2 lists the fairness commitments. The one that is easiest to get wrong:

**Do not use global average pooling before Approach A's FC head.** GAP discards all spatial
information, which for a localisation task is close to fatal. Flatten a small spatial grid instead
so the FC layer can still read position. A GAP-based Approach A would fail for a reason that has
nothing to do with the regression-vs-heatmap question, and the comparison would be worthless.

Also required by the spec (§5.1 hint) and easy to forget: **write down your prediction before
running the experiments**, then report whether it held. Put it in `state/discoveries.md`, dated.

---

## Sources

- *Data Efficient Training of a U-Net Based Architecture for Structured Documents Localization* —
  arXiv:2310.00937 (SDL-Net). Resolution, heatmap format, loss, extraction, IoU metric.
- Wang et al., *Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression*, ICCV 2019 —
  the foreground/background imbalance problem and its fix.
- Luo et al., *Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation*
  (arXiv:2012.15175) — scale-adaptive σ; the case that fixed σ is a simplification.
- SmartDoc 2015 (ICDAR) — the standard document localisation benchmark.
- `DocsaidLab/DocAligner` — an open-source document four-corner predictor, useful as a reference
  point for what production systems do.
