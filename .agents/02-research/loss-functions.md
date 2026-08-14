# Loss Functions for Document Enhancement

Background for ADR-006. The decision is there; this is the reasoning and the literature.

Spec §3.2 raises this as a question — "A standard pixel-wise loss like Mean Squared Error is known
to produce **blurry outputs** … Do you have any idea?" — and `[REQ-45]` makes the *comparison* a
graded deliverable. So this is not just about picking a winner; the analysis itself is the product.

---

## Why L2 blurs — the actual mechanism

Worth being able to state precisely, because it is the most likely question at the presentation.

Given a degraded input, several clean images are plausible — a stroke edge could be at pixel 100 or
101. The network cannot resolve which. Under **L2**, the minimising output is the **conditional
mean** over plausible targets: averaging the candidates costs less than committing to one and being
wrong. Averaging two edge positions produces a soft ramp. The result is systematic smoothing,
strongest exactly where uncertainty is highest — which for a document is *at every text edge*.

Under **L1**, the minimiser is the **conditional median**. The median of a set of candidate edge
positions is an actual edge position, not a blend of them. So L1 commits, and edges stay sharp.

This is why L1 is standard in modern restoration work and why the spec's hint names it first.

**The measurement trap that follows:** PSNR is a monotone function of MSE. A model trained on L2 is
*directly optimising the thing PSNR measures*, so it will often score highest on PSNR while looking
visibly worse. **Expect this in the ablation, and say so in the report** — noticing it is exactly
the insight `[REQ-45]` rewards. Judge sharpness by SSIM, by the OCR result, and by looking at the
images.

---

## The candidates

### L2 / MSE
Smooth gradients, easy optimisation, directly maximises PSNR. Blurs. Present in the ablation as
the spec's named straw man and as the PSNR reference point.

### L1 / MAE
Sharper edges, robust to outliers, stable colour reproduction. Ignores structure entirely: it
scores every pixel independently, so it has no notion of local contrast or texture coherence. A
uniformly slightly-wrong image and a locally-very-wrong image can score the same.

### SSIM / MS-SSIM
Perception-motivated. Compares local **luminance**, **contrast** and **structure** in a sliding
window, rather than raw pixel differences. Because it operates on local statistics, it rewards
preserving *relative* structure — which for a document means keeping the contrast between stroke
and paper, i.e. legibility.

**MS-SSIM** evaluates SSIM across five scales with fixed weights, so it captures both fine stroke
detail and page-scale illumination structure. That multi-scale property is a good match for this
task, where the two failure modes (blurred text, residual shadow) live at opposite scales.

Weakness: SSIM is insensitive to uniform intensity shifts. A model trained on MS-SSIM alone can
drift in overall brightness — a bad failure for a document that should be white. **This is why the
mixture exists**, not because mixing is generically good: the L1 term anchors absolute intensity
while MS-SSIM handles structure.

### Sobel / gradient loss
Apply a fixed Sobel operator to prediction and target, penalise the L1 difference of the gradient
magnitudes. Directly targets "text legibility lives in the edges" (spec §3.2 hint).

Risk: the operator cannot distinguish a text edge from a **noise** edge or a JPEG block boundary.
With a large weight it will faithfully reconstruct the sensor noise your degradation pipeline added.
Keep λ small (~0.1).

### Perceptual / VGG / LPIPS — **unavailable**
The standard modern answer for perceptual sharpness. **Banned by `[CON-02]`**: all variants require
pretrained network weights. Worth one sentence in the report's "approaches we could not try" —
it demonstrates awareness of the field while respecting the constraint.

### Adversarial — **out of scope**
Would give the sharpest text. Adds a discriminator, training instability and a second failure mode,
for a project graded on PSNR/SSIM and OCR. `[CON-10]`.

---

## The mixed loss — Zhao et al. 2017

**Zhao, Gallo, Frosio & Kautz**, *Loss Functions for Image Restoration with Neural Networks*,
IEEE Transactions on Computational Imaging 3(1):47–57, 2017. arXiv:1511.08861.

The reference for this recipe. It compares L1, L2, SSIM, MS-SSIM and mixtures across denoising,
demosaicing and super-resolution, and finds that a weighted **MS-SSIM + L1** mixture performs best
on both distortion and perceptual measures. The widely-cited weight is **α = 0.84**.

⚠️ **Read §5 of the paper before implementing.** The paper's mixed loss applies a **Gaussian
weighting** to the L1 term, derived from the coarsest MS-SSIM scale:

```
L_mix  =  α · L_MS-SSIM  +  (1 − α) · G_σM · L1
```

Most public reimplementations drop the `G_σM` factor and use plain L1. The simplified form works
well and is acceptable here — **but implement whichever you choose knowingly and state which in the
report.** Citing the paper for a formulation you did not implement is the kind of thing that gets
caught in a viva.

**Does α = 0.84 transfer to documents?** Unknown — `[ASM-04]`. It was tuned on natural images;
documents are high-contrast and near-bimodal, with large flat white regions where SSIM behaves
differently. ADR-006 prescribes a cheap sweep over α ∈ {0.7, 0.84, 0.95} after L-C trains.

---

## Practical notes

**MS-SSIM minimum size.** Five scales with an 11×11 window requires images ≥ `(11−1)·2⁴+1 = 161` px.
512 is comfortable. If anything ever runs smaller, reduce the number of scales rather than letting
it throw.

**Numerical stability.** The SSIM denominator involves local variance, which is ≈0 on a flat patch —
and a document's blank margin is exactly that. The stabilising constants C1 and C2 are load-bearing.
Do not simplify them away, and watch for NaN if you do.

**Mixed precision.** Cast the MS-SSIM computation to `float32` explicitly under AMP. Its products,
divisions and small-variance terms are not fp16-safe and will produce NaN — a confusing failure
because the rest of training looks fine until the loss becomes NaN.

**Sign convention.** SSIM and MS-SSIM are *similarities* in `[0,1]`, higher-is-better. The loss is
`1 − MS-SSIM`. Getting this backwards trains a model to maximally destroy structure, which is
obvious in the output but has cost people days.

**Loss scale.** L1 on `[0,1]` images gives values ~0.01–0.1; `1 − MS-SSIM` gives ~0.0–0.5. They are
roughly comparable, so α ≈ 0.84 is not fighting a scale mismatch. If you add the Sobel term, check
its magnitude before choosing λ — gradient magnitudes can be much larger than pixel differences.

---

## What to report

`[REQ-45]` wants the comparison, not just the winner:

- One table: each loss variant × (PSNR, SSIM, and if available the real-photo OCR result).
- One figure: the **same input** enhanced by all four models, side by side, zoomed on a text region
  so the sharpness difference is visible.
- The PSNR/SSIM disagreement, explained (the L2 model probably wins PSNR and loses on looks).
- Whether the Sobel term helped, hurt, or did nothing — **a null result is a result**.
- The α sweep, if it moved anything.

---

## Sources

- Zhao, Gallo, Frosio, Kautz (2017), *Loss Functions for Image Restoration with Neural Networks*,
  IEEE TCI 3(1):47–57 — arXiv:1511.08861. Source of the mixed loss and α=0.84.
- Wang, Simoncelli, Bovik (2003), *Multiscale structural similarity for image quality assessment* —
  MS-SSIM and its scale weights `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`.
- Wang, Bovik, Sheikh, Simoncelli (2004), *Image quality assessment: from error visibility to
  structural similarity* — original SSIM.
