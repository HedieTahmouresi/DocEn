# ADR-005 — Enhancement Network Architecture

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** Medium

## Context

`[REQ-19]` (§3.1) requires an encoder-decoder with skip connections, built from scratch, from
standard layers. `[CON-01]` forbids importing a ready-made U-Net; `[CON-02]` forbids pretrained
weights; `[CON-04]` forbids dropout and other explicit regularisation in the first version.

The task: map a degraded, already-rectified 512×512 document to a clean scan. Two demands pull in
opposite directions — **global context** (an illumination gradient or a shadow spans the whole
page, so the receptive field must be large) and **fine detail** (text strokes are 1–3 px wide and
must survive). That tension is exactly what a U-Net's skip connections resolve, which is why the
spec points at them.

## Decision

**A hand-written U-Net: 4 downsampling levels, symmetric decoder, concatenating skip connections.**

Starting configuration (tune the widths freely; the *structure* is the decision):

```
input  3 × 512 × 512
 enc1   64 × 512 × 512   ── skip ──────────────────────────┐
 pool          256                                          │
 enc2  128 × 256 × 256   ── skip ────────────────────┐      │
 pool          128                                    │      │
 enc3  256 × 128 × 128   ── skip ─────────────┐       │      │
 pool           64                             │       │      │
 enc4  512 ×  64 ×  64   ── skip ──────┐       │       │      │
 pool           32                      │       │      │      │
 bottleneck 512 × 32 × 32               │       │      │      │
 up                                     │       │      │      │
 dec4  512 ×  64 ×  64  ◄── concat ─────┘       │      │      │
 dec3  256 × 128 × 128  ◄── concat ─────────────┘      │      │
 dec2  128 × 256 × 256  ◄── concat ────────────────────┘      │
 dec1   64 × 512 × 512  ◄── concat ───────────────────────────┘
 head    3 × 512 × 512   1×1 conv → sigmoid
```

**Block:** `Conv3×3 → BatchNorm → ReLU`, twice per level. The canonical U-Net block.

**Downsampling:** `MaxPool2d(2)`. Named in the spec's own layer list.

**Upsampling:** start with `ConvTranspose2d(stride=2)`. If checkerboard artifacts appear in the
output — likely on text — switch to `Upsample(scale_factor=2, mode='bilinear') → Conv3×3`. Both are
named in spec §3.1; the switch is an ordinary engineering call, just record which you used.

**Skips:** concatenation (not addition). Canonical, and it lets the decoder weigh encoder detail
against decoder semantics rather than forcing a sum.

**Head:** 1×1 conv to 3 channels, `sigmoid`, output in `[0,1]` (ADR-009).

**Receptive field check:** at the bottleneck each unit sees a 32×32 grid position, i.e. a ~16×
downsampled view of the full page — enough to represent a page-wide illumination gradient. This is
the justification for 4 levels rather than 3. Do not go to 5 without a measured reason; the gain is
small and the parameter count grows fast.

### BatchNorm and `[CON-04]`

**BatchNorm is used, deliberately, and this reading is recorded so it can be defended.**

`[CON-04]` forbids "dropout layers or other explicit regularization techniques". BatchNorm is a
normalisation/optimisation layer: its purpose is conditioning the optimisation problem, and it is
part of the standard convolutional block the spec's own layer list implies. Its mild regularising
side-effect is incidental, not its function. Dropout, weight decay, label smoothing and mixup are
*explicit* regularisers — techniques whose entire purpose is to regularise. That is the distinction
being drawn.

**Be ready to state this at the presentation.** If a TA rules otherwise, the fallback is GroupNorm
or InstanceNorm, which carry no regularisation interpretation at all. That would be a small,
localised change — note it as a contingency, not a task.

**Not negotiable in the same breath:** `weight_decay=0.0` in the optimizer for Phase 04 and
Phase 06 models. Weight decay *is* an explicit regulariser and is unambiguously covered by
`[CON-04]`. Note that `AdamW` defaults to `0.01` — use `Adam`, or set it explicitly.

## Consequences

**Good.** Canonical, defensible, cheap to explain. Bottleneck retains 32×32 spatial extent, so
global illumination context survives. Skips preserve stroke detail. Roughly 8M parameters at the
widths above — comfortable on a T4 at 512 with AMP, batch 8–16.

**Costs.** ~8M parameters is more than this dataset strictly needs; watch for overfitting to the
synthetic distribution (though on-the-fly generation is a strong counterweight). Memory at
512×512 with 64 channels at full resolution is the dominant cost — if OOM on Colab, halve the
base width to 32 before reducing batch size.

## Explicitly rejected (would be over-engineering — `[CON-10]`)

- **Attention-gated skips / transformer blocks.** The research report notes the "semantic gap" in
  plain U-Net skips is a real phenomenon in the literature, then concludes that addressing it here
  is "extreme overengineering for this specific project parameter." Agreed.
- **Diffusion (DocDiff) or contrastive frameworks (DocNLC).** Far out of scope.
- **GAN/adversarial refinement.** Would sharpen text, but adds a discriminator, training
  instability, and a second failure mode, for a project graded on PSNR/SSIM and OCR.
- **Perceptual/VGG loss.** Banned outright by `[CON-02]` — it requires pretrained weights.
- **Deeper (5–6 level) or wider nets.** No evidence they are needed. Revisit only if Phase 04
  diagnoses *underfitting* (training metrics themselves poor), per `05-skills/training-diagnostics.md`.

## Permitted variations (implementation agent's judgment — `[REC]`)

- **Residual formulation** (predict the residual `output = sigmoid(input + f(input))` rather than
  the image directly). Often helps restoration converge faster since the input and target are
  already similar. **Try only after the plain version trains and passes its gate**, and log it as a
  separate experiment so the comparison is clean.
- Channel widths, and whether the bottleneck doubles to 1024.
- Bilinear-upsample vs transposed conv, as above.
- Kernel size in the first layer (5×5 or 7×7 for a larger initial receptive field).

## Reuse for corner detection

`[REQ-30]` says Approach B may "reuse your encoder-decoder machinery from Section 3." Build this as
a **configurable module** parameterised by output channels and final activation:
- Enhancement: `out_channels=3`, `sigmoid`
- Approach B: `out_channels=4`, `sigmoid` (heatmaps in `[0,1]`)
- Approach A: the **encoder half only**, plus a flatten + fully-connected head (ADR-007)

One tested implementation, three uses. See `03-spec/model-specs.md`.
