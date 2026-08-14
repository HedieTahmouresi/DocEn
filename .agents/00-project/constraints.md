# Constraints Register — Things You Must **Not** Do

**Source of authority:** `Document Scanning Enhancement.md`. Quoted verbatim with citations.

These are the prohibitions. Violating one is worse than failing to build a feature: it can
invalidate the work. Several are easy to violate *by accident* through a convenient import, so
each entry lists the specific traps.

---

## `[CON-01]` — No pre-designed architectures

> *Spec §3.1 🚨:* "We will not use pre-designed architectures (like importing a ready-made U-Net)
> or pre-trained weights."
> *Spec §5 🚨:* "As in Section 3, do not use pre-trained weights or dropout layers here — first
> versions of both corner detectors are built clean."

**Banned:** `segmentation_models_pytorch`, `monai` networks, `torchvision.models`,
`timm`, any `UNet` class copied wholesale from a repo, Keras Applications.

**Required instead:** the encoder, decoder, skip connections and heads are written by hand from
`nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.ConvTranspose2d`/`nn.Upsample`,
`nn.Linear`.

**Trap:** "I'll just use torchvision's ResNet encoder without loading weights" — still banned.
It is a pre-designed architecture regardless of initialisation.

---

## `[CON-02]` — No pre-trained weights, anywhere

> *Spec §3.1 🚨 and §5 🚨* (quoted above).

**Banned:** ImageNet backbones, and — the trap most people miss — **any perceptual loss**.
VGG-perceptual loss, LPIPS, and DISTS all load pretrained networks. They are therefore
unavailable no matter how well they would suit image restoration.

**Consequence for loss design:** the loss menu is limited to pixel losses (L1/L2), structural
losses computed from first principles (SSIM/MS-SSIM), and gradient losses (Sobel). This is
exactly the menu the spec hints at in §3.2. See `02-research/loss-functions.md`.

**Also banned:** ImageNet mean/std normalisation constants. There is no pretrained network to
match, so normalisation statistics must be derived from this project's own data (ADR-009).

---

## `[CON-03]` — No third-party libraries for the image transformations

> *Spec §4 🚨 Implementation note:* "You are **not allowed** to use any third-party libraries to
> handle transformations. You **MUST** use the techniques learned in the course and *OpenCV
> functions* to build the degradation pipeline."

**Banned in the degradation pipeline:** `albumentations`, `imgaug`, `kornia.augmentation`,
`torchvision.transforms` (for the degradations), `Pillow` filters, `scipy.ndimage` warping.

**Allowed:** `cv2.*` and `numpy`. Everything in the six-step pipeline of `[REQ-34]` — the
homography, the resampling, the gradient masks, the shadow polygons, the blur, the noise, the
JPEG round-trip — is written with OpenCV and NumPy.

**Scope note.** This ban is written specifically about *transformations* (spec §4). It is not a
blanket ban on all libraries. `torch`, `numpy`, `matplotlib`, `scikit-image` (for metric
cross-checks), `pytesseract` (OCR evaluation) and `kornia.geometry` (bonus differentiable warp,
which the spec itself recommends in §7) are all fine.
**But:** where a grey area exists, prefer writing it yourself. See ADR-010, which chose to
implement SSIM/MS-SSIM by hand rather than import them, for exactly this reason.

---

## `[CON-04]` — No dropout or other explicit regularisation in first versions

> *Spec §3.1 🚨:* "in this section, you should not use any dropout layers or other explicit
> regularization techniques. The emphasis is on good model training on the dataset and the
> original architecture design."
> *Spec §5 🚨:* "do not use pre-trained weights or dropout layers here — first versions of both
> corner detectors are built clean. Regularization comes in Section 6."

Applies to **all three networks** (enhancement, corner-A, corner-B) in Phases 04 and 06.

**Traps:**
- `weight_decay` in the optimizer **is** explicit regularisation. Set `weight_decay=0.0`.
  (Note `torch.optim.Adam` defaults to 0, but `AdamW` defaults to 0.01 — do not use `AdamW`
  for these runs without setting it to 0.)
- Label smoothing, mixup, cutout, stochastic depth, `DropBlock`, `nn.Dropout2d` — all banned here.
- Early stopping is a grey area. It is model *selection*, not a regularisation layer, and the spec
  requires validation-based monitoring anyway (`[REQ-21]`). Using "best validation checkpoint" is
  fine and expected; document it.
- **BatchNorm is permitted.** It is a normalisation/optimisation layer, not an explicit regulariser
  in the sense meant here, and the spec's own layer list is standard-CNN. ADR-005 records this
  reading and its justification — be ready to defend it at the presentation.
- Data augmentation is not only permitted, it is the entire project (`[REQ-07]`).

**Phase 07 lifts this constraint** — but only there, and only as a *measured comparison*
(`[REQ-38]`, `[REQ-39]`). The un-regularised models must be kept and reported alongside.

---

## `[CON-05]` — No flips

> *Spec §4.1:* "We will avoid flipping: mirrored text is not something a document scanner should
> ever learn to 'restore.'"

No horizontal or vertical flips. Note this also constrains rotation: a 180° rotation is not a flip
and is geometrically legitimate, but produces upside-down text, which is equally not something the
scanner should learn to fix. Keep in-plane rotation within the range real photos actually exhibit
(ADR-003 and `03-spec/synthetic-generator-spec.md`).

---

## `[CON-06]` — Never train on the real photos

> *Spec §1.1:* "These photos are reserved for testing only."
> *Spec §2.3:* "Never train on them — and never run the degradation pipeline on them: they arrive
> degraded by reality."
> *Spec §1.1:* "Reference scans are for evaluation only and must never be used for training."

This includes: no fine-tuning on them, no using them for validation-based model selection, no
including them in any training split.

**Grey area, handled explicitly.** ADR-004 permits *measuring statistics* of the real photos
(their corner-quad geometry, brightness distribution, blur level) in order to calibrate the
synthetic generator's parameter ranges. The spec sanctions this in §1.1 ("whatever degradations you
see — shadows, blur, colour casts, perspective distortion — are exactly what your synthetic
pipeline in Section 4 must reproduce") and §4.4. **But** the calibration must deliberately *widen*
the ranges beyond the observed statistics, because the graded evaluation happens on a hidden set
(`[REQ-49]`). Fitting the generator tightly to your own 10–15 photos is a subtle form of
overfitting. See `02-research/sim2real-playbook.md`.

---

## `[CON-07]` — Never touch the synthetic test split until final evaluation

> *Spec §3.2:* "your synthetic test set stays untouched until Section 3.3."
> *Spec §2.3:* "the test set is held out and touched once, at the end, to report final numbers."

No test-set-based model selection, no peeking to decide hyperparameters, no "just to check".
Validation exists for that. See `05-skills/eval-integrity.md`.

---

## `[CON-08]` — No corner coordinates fed into the enhancement network

> *Spec §1.3 🚨:* the enhancement network's "input is the degraded document warped back to a flat
> rectangle; its target is the original clean scan."

The enhancement network is a pure image-to-image module. Once the image is rectified, the original
corner coordinates carry no further information. The research report reaches the same conclusion
independently and calls the alternative "an overengineering fallacy." Do not add coordinate
conditioning, positional priors, or a second input branch.

---

## `[CON-09]` — Do not deviate from the specified degradation order

> *Spec §4.3* gives the exact six-step sequence (`[REQ-34]`).

The order is physically motivated: a real camera degrades the whole scene *after* the geometry is
established, and compression happens last. Reordering (for example, applying noise after JPEG, or
blurring before the perspective warp) changes the statistics of the resulting images and departs
from a written requirement.

You may add *parameters* within each step and you may randomise whether an optional sub-effect
fires (e.g. shadow present or absent) — that is `[REQ-36]`. You may not reorder the steps or drop
one.

---

## `[CON-10]` — Scope boundary: this is a course project, not a research contribution

Not a spec quotation — a decision recorded here because it is enforced like a constraint.

The literature contains diffusion-based document enhancement (DocDiff), contrastive
representation frameworks (DocNLC), attention-gated skip connections, transformer U-Nets, and
adversarial refinement. **None of these are in scope.** The research report reaches the same
conclusion: implementing them "falls into the realm of extreme overengineering for this specific
project parameter."

Before adding *any* component beyond what `03-spec/model-specs.md` describes, load
`05-skills/scope-guard.md` and follow its procedure.

---

## Quick self-check before any commit

- [ ] No import of a pre-built network or pretrained weights?
- [ ] Degradation pipeline still OpenCV + NumPy only?
- [ ] `weight_decay == 0` and no dropout in Phase 04 / Phase 06 models?
- [ ] No flips in the augmentation?
- [ ] Real photos untouched by training and by the degradation pipeline?
- [ ] Synthetic test split untouched (unless you are in Phase 05 / 06 final evaluation)?
- [ ] Degradation steps still in the §4.3 order?
