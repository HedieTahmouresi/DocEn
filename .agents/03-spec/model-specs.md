# Model Specifications

Three networks, one shared backbone module. Architecture decisions are ADR-005 (enhancement),
ADR-007 (corner approaches) and ADR-008 (heatmaps).

**Constraints throughout:** `[CON-01]` no pre-built architectures · `[CON-02]` no pretrained
weights · `[CON-04]` no dropout and `weight_decay=0` in Phases 04 and 06.

All three operate at **512×512** (ADR-002).

---

## 0. Shared design: one configurable encoder-decoder

`[REQ-30]` says Approach B may "reuse your encoder-decoder machinery from Section 3." Build it once,
parameterised, and use it three ways. One implementation to test, debug and explain.

```
Encoder(in_ch=3, base=64, levels=4)          -> features, skips
Decoder(base=64, levels=4, out_ch, out_act)  -> image or heatmaps
```

| Network | Uses | `out_ch` | `out_act` |
|---|---|---|---|
| Enhancement | Encoder + Decoder | 3 | sigmoid |
| Corner Approach B | Encoder + Decoder | 4 | sigmoid |
| Corner Approach A | **Encoder only** + FC head | 8 | sigmoid |

Approach A sharing the encoder is a fairness requirement, not just convenience (ADR-007 §2): any
measured difference must come from the *output representation*, not from one model having a better
backbone.

---

## 1. Enhancement network

**Requirement:** `[REQ-19]` — encoder-decoder with skip connections, from scratch, standard layers.

```
input                   3 × 512 × 512   (standardised, ADR-009)
├─ enc1  DoubleConv     64 × 512 × 512  ──── skip 1 ────────────────┐
│  pool  MaxPool2d(2)                                                │
├─ enc2  DoubleConv    128 × 256 × 256  ──── skip 2 ──────────┐      │
│  pool                                                        │      │
├─ enc3  DoubleConv    256 × 128 × 128  ──── skip 3 ───┐       │      │
│  pool                                                 │       │      │
├─ enc4  DoubleConv    512 ×  64 ×  64  ──── skip 4 ┐   │       │      │
│  pool                                             │   │       │      │
├─ bott  DoubleConv    512 ×  32 ×  32              │   │       │      │
│  up    ConvTranspose2d(2)                         │   │       │      │
├─ dec4  DoubleConv    512 ×  64 ×  64  ◄─ concat ──┘   │       │      │
├─ dec3  DoubleConv    256 × 128 × 128  ◄─ concat ──────┘       │      │
├─ dec2  DoubleConv    128 × 256 × 256  ◄─ concat ──────────────┘      │
├─ dec1  DoubleConv     64 × 512 × 512  ◄─ concat ─────────────────────┘
└─ head  Conv2d(64, 3, k=1) → Sigmoid
                         3 × 512 × 512   output in [0,1]
```

**`DoubleConv(in, out)`** = `Conv3×3(pad=1) → BatchNorm2d → ReLU` twice.

- **BatchNorm is deliberate** and permitted under `[CON-04]` — ADR-005 records the reasoning and the
  GroupNorm fallback. Be ready to defend it.
- **Skips concatenate**, not add.
- **Upsampling:** `ConvTranspose2d(stride=2)` to start. If checkerboard artifacts appear on text,
  switch to `Upsample(bilinear) → Conv3×3`. Both are named in spec §3.1; record which you used.
- **Bottleneck at 32×32** carries page-scale illumination context — the justification for 4 levels.

~8M parameters at these widths. If Colab OOMs, halve `base` to 32 before reducing batch size —
capacity is less valuable here than batch statistics for BatchNorm.

`[REC]` **Residual formulation** (`output = sigmoid(x + f(x))`) often speeds convergence since input
and target are already similar. Try only **after** the plain version passes its gate, and log it as
a separate experiment.

---

## 2. Corner Approach A — direct coordinate regression

**Requirement:** `[REQ-30]` — "A CNN encoder followed by fully connected layers that output 8
numbers: the normalized (x, y) coordinates of the four corners."

```
input                   3 × 512 × 512
└─ shared Encoder (4 levels) →   512 ×  32 ×  32
   ├─ extra pool/stride       →  512 ×   8 ×   8      reduce before flatten
   ├─ flatten                 →  32768
   ├─ Linear(32768, 512) → ReLU
   ├─ [Phase 07 only: Dropout here]
   ├─ Linear(512, 256)   → ReLU
   └─ Linear(256, 8)     → Sigmoid
                              8 values, normalised coords in [0,1]
```

Output ordering: `[x0,y0, x1,y1, x2,y2, x3,y3]` = TL, TR, BR, BL
(`00-project/conventions.md` §1).

### Two things that determine whether the comparison is honest

**Do not use global average pooling before the FC head.** GAP discards all spatial information,
which for a localisation task is close to fatal — Approach A would then fail for a reason unrelated
to the regression-vs-heatmap question, and the whole comparison would be worthless. Reduce to a
small spatial grid (8×8) and flatten it, so the FC layer can still read position. This is ADR-007's
central fairness commitment.

**Do not sort the outputs to enforce ordering.** Regression models can emit corners in an
inconsistent order; that is a real, reportable failure mode. Sorting hides it and breaks on rotated
pages.

**Loss:** L1 on normalised coordinates. The spec permits L1 or L2; L1 is more robust to outliers and
is the better-faith choice.

~17M parameters, dominated by the first Linear layer. That is expected — it is exactly the
inefficiency the architectural critique predicts, and it is worth mentioning in the report.

---

## 3. Corner Approach B — heatmap regression

**Requirement:** `[REQ-30]` · **Design:** ADR-008

```
input                   3 × 512 × 512
└─ shared Encoder + Decoder (identical to enhancement)
   └─ head  Conv2d(64, 4, k=1) → Sigmoid
                         4 × 512 × 512   heatmaps in [0,1]
```

Channel `c` ↔ corner `c`, in TL, TR, BR, BL order.

### Targets
- 2D Gaussian, **peak exactly 1.0** at the corner, σ = **8 px** at 512 (`[ASM-05]`, sweep {4,8,12}).
- **Rendered in a ±3σ window and pasted** — never evaluated over the full 512² canvas. ~100× cheaper
  for a numerically identical result (ADR-008), and load-bearing for CPU throughput on Colab.
- Near the border, **clip the window; never shift it** — shifting moves the peak and corrupts the
  label.

### Loss
Plain **MSE** first — the spec's default reading of "a pixel-wise loss", and proven to train on this
task.

**Known risk, so you recognise it rather than debug it from scratch:** the Gaussian covers ~0.7% of
pixels, so the loss is background-dominated. *Signature:* heatmaps collapse toward zero, loss drops
fast then plateaus, argmax returns noise.
**Pre-approved response** (log as an experiment, no escalation): foreground-weighted MSE, pixel
weight `1 + w·target`, `w ≈ 10–50`. Adaptive Wing Loss needs escalation.

### Coordinate extraction
1. `argmax` per channel → integer peak.
2. **Local soft-argmax** in an 11×11 window around it → sub-pixel coordinate.

Global soft-argmax is rejected: this project deliberately trains on cluttered backgrounds, so
secondary peaks are an *expected* output, and a whole-map expectation would land the corner between
two candidates.

Also record **peak activation as a confidence score** (SDL-Net does the same). Free, and it
separates "wrong and unsure" from "wrong and confident" in failure analysis.

---

## 4. Where dropout goes in Phase 07 — and nowhere before

`[CON-04]` forbids all of this until Phase 07. `[REQ-38]` then requires it in both models.

| Model | Placement | Spec guidance |
|---|---|---|
| Approach A | Between the FC layers | §6: "the fully connected layers are the classic place" |
| Enhancement | **Bottleneck only**, `Dropout2d` | §6: "experiment with where in the architecture it helps" |
| Approach B | Bottleneck, and optionally deepest encoder levels | same |

**Bottleneck-only for the encoder-decoders** is the reasoned starting point: dropout in early
high-resolution layers destroys the fine spatial features that thin text strokes depend on, whereas
bottleneck features are compressed and semantic. For Approach B specifically, dropping bottleneck
activations forces the network to infer a corner from global page geometry rather than from one
memorised local texture — which is exactly the domain-overfitting attack `[REQ-39]` asks about.

Rate: start at `p=0.2`, sweep `{0.1, 0.2, 0.3}` if time allows.

**Keep the Phase 04/06 checkpoints.** `[REQ-38]` requires reporting *the difference*, so the
un-regularised models must survive as the comparison arm.

---

## 5. Initialisation

`[CON-02]` means every weight is initialised from scratch. Use **Kaiming/He normal** for conv layers
followed by ReLU (`fan_out` mode), zeros for biases, and default init for BatchNorm (weight 1, bias
0).

Do not leave PyTorch's default `Linear` init on the Approach A head without checking it — with a
32768-dimensional input, poor scaling there can stall training and be misread as "regression doesn't
work", which would corrupt the `[REQ-30]` comparison.

---

## 6. Parameter budget

| Model | Approx. params | Notes |
|---|---|---|
| Enhancement | ~8M | `base=64`, 4 levels |
| Approach A | ~17M | dominated by `Linear(32768, 512)` |
| Approach B | ~8M | same skeleton as enhancement, 4 output channels |

All comfortable on a T4 at 512×512 with AMP. On the MX330 (4 GB), expect batch 2–4 and use it only
for smoke tests (ADR-001).

---

## 7. Interface contract

Keep these stable — the pipelines, evaluation and the bonus chain all depend on them.

```
EnhancementNet.forward(x)      x: (N,3,512,512) standardised float32
                            -> (N,3,512,512) float32 in [0,1]

CornerRegNet.forward(x)        x: (N,3,512,512) standardised float32
                            -> (N,8) float32 in [0,1], normalised coords, TL,TR,BR,BL

CornerHeatmapNet.forward(x)    x: (N,3,512,512) standardised float32
                            -> (N,4,512,512) float32 in [0,1]

extract_corners(heatmaps)      heatmaps: (N,4,512,512)
                            -> coords: (N,4,2) float32, absolute px, sub-pixel
                               conf:   (N,4)   float32, peak activation
```

Docstrings state shape, dtype, range and colour space for every image or coordinate argument
(`00-project/conventions.md` §10).
