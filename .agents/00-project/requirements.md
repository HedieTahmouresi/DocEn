# Requirements Register

**Source of authority:** `Document Scanning Enhancement.md` (the official course specification).
Every entry below is traceable to a numbered section of that document. Quoted text is verbatim.

> **The citation rule.** If a claimed requirement does not appear in this file with a section
> citation, **it is not a requirement.** It is at most a recommendation. Do not invent
> requirements. Do not promote statements from the research report
> (`Document Scanner Implementation Plan.md`) into this file — that document is advisory.

Prohibitions live in a separate file: `constraints.md` (`[CON-nn]`).

---

## A. Data preparation

### `[REQ-01]` — Provided clean scans are the ground-truth targets
*Spec §1.1.* The teaching staff provide "a collection of clean, flat, high-resolution document
scans. These are your ground truth targets for training."

### `[REQ-02]` — Collect 10–15 real smartphone photos of previously unseen documents
*Spec §1.1.* "Collect 10–15 real smartphone photos of your own documents — pages your model has
never seen in any form." Purpose: measure generalisation to *new documents*, not just new photos
of familiar ones.

### `[REQ-03]` — Produce a commercial reference scan for each real photo
*Spec §1.1.* For each photo, "also produce a reference scan of the same document using a scanning
app (CamScanner, Adobe Scan, or your phone's built-in document scanner)."
Explicitly: "This is not ground truth — it is a commercial baseline to compare yourself against.
Reference scans are for evaluation only and must never be used for training."

### `[REQ-04]` — Real photos must be diverse
*Spec §1.1 hint.* Vary lighting (daylight, warm lamp, harsh overhead, shadow across page),
viewpoint (angles, distances, rotations), background (desk, carpet, cluttered table), camera
behaviour (shake, imperfect focus), and document type (dense text, sparse text, a figure, a
coloured logo).

### `[REQ-05]` — Annotate four page corners on every real photo, in consistent order
*Spec §1.2.* Use RoboFlow **keypoint annotation** mode (CVAT or Label Studio also acceptable).
"For every photo, place four keypoints on the page corners in a consistent order (top-left,
top-right, bottom-right, bottom-left)."
*Spec §1.2 hint:* "Consistent corner ordering is not optional — if your labels mix up top-left
and bottom-right across images, both the evaluation and the rectification in the bonus part will
silently break."

### `[REQ-06]` — Share the RoboFlow project link with the teaching assistants
*Spec §2.1.* "Upload the link of your Roboflow project … to the designated location in Google
Sheets." Project must be public or TA-accessible. **This is a human action, not an agent action.**

### `[REQ-07]` — Generate training pairs synthetically; no hand annotation of training data
*Spec §1.3.* "You never need to annotate the training set at all." Per sample: take a clean scan
and a random background; choose four random target points defining a homography; warp with
`cv2.getPerspectiveTransform` and `cv2.warpPerspective`; degrade the result. The four chosen
points **are** the corner labels.

### `[REQ-08]` — One generator serves both tasks
*Spec §1.3.* "Because you know the homography, you can warp the degraded photo back to obtain a
perfectly aligned (degraded input, clean target) pair for the enhancement network. The label
generator and the data generator are the same function."

### `[REQ-09]` — The enhancement network operates on the rectified crop
*Spec §1.3 🚨 Implementation.* "The enhancement network operates on the **rectified crop**, not on
the raw photo. Its input is the degraded document warped back to a flat rectangle; its target is
the original clean scan. This decouples the two mandatory tasks: the enhancement network and the
corner detector are trained and evaluated independently, and are only chained together in the
bonus part."

---

## B. Preprocessing

### `[REQ-10]` — Parse the keypoint annotation export into ordered (4,2) arrays
*Spec §2.2 step 1.* Export RoboFlow keypoints (COCO keypoint JSON recommended), parse with
`pycocotools` or plain `json`, "store the four corners of every real photo as an ordered array of
shape (4, 2)."

### `[REQ-11]` — The synthetic generator must be wrapped in an on-the-fly Dataset
*Spec §2.2 step 2.* "Your Dataset class should generate samples **on the fly**: each
`__getitem__` call composites a fresh (degraded input, clean target, corner coordinates) triple.
This gives you a practically infinite training set without ever writing images to disk."
→ See `[DEC]` ADR-003 for how this interacts with the frozen evaluation sets of `[REQ-15]`.

### `[REQ-12]` — Resize images and their corner labels together
*Spec §2.2 step 3.* "When resizing an image, its corner coordinates must be **scaled by the same
factors**. A corner label that is not transformed together with its image is a wrong label."

### `[REQ-13]` — Normalise pixels and corner coordinates
*Spec §2.2 step 4.* Scale pixel values (e.g. `/255.0` to `[0,1]`). "**Normalize the corner
coordinates as well** (divide by image width and height) so that they live in [0, 1] — this makes
the corner detection task resolution-independent."

### `[REQ-14]` — Split by source scan, 80/10/10
*Spec §2.3.* "Split by source scan, not by generated sample — two degraded versions of the same
page must never end up on different sides of a split. A reasonable division of the scan collection
is 80% / 10% / 10%." Validation is for monitoring and model selection; "the test set is held out
and touched once, at the end, to report final numbers."

### `[REQ-15]` — Freeze the validation and test sets
*Spec §2.3.* "Generate the validation and test samples once with a fixed random seed (or write
them to disk), so that every epoch, and every model you compare, is scored on identical images."
Rationale given in spec: otherwise "your validation curve would then measure the dice as much as
the model."

### `[REQ-16]` — Real photos form a fourth, separate evaluation set
*Spec §2.3.* Used in its entirety. Preparation:
- **For enhancement:** rectify each photo using the annotated corners, then resize and normalise
  exactly as for synthetic inputs; resize the reference scan to the same size so the two are
  directly comparable.
- **For corner detection:** input is the raw photo, resized and normalised, with annotated corners
  scaled by the same factors.

### `[REQ-17]` — Both mandatory tasks share the same split
*Spec §2.3.* "Both mandatory tasks share this split: the same source scans, the same held-out
photos."

### `[REQ-18]` — Verify the data pipeline before training
*Spec §2.4.* Test the DataLoader loads without errors. "Visualize a few (input, target) pairs side
by side and overlay the corner labels on the composited photos to check that everything is
aligned."

---

## C. Task 1 — Enhancement network

### `[REQ-19]` — Design an encoder-decoder with skip connections, from scratch
*Spec §3.1.* "The encoder progressively downsamples the degraded input to capture context … and
the decoder progressively upsamples back to a full-resolution clean image." Standard layers:
`Conv2D`, `MaxPooling2D`, `UpSampling2D` (or transposed convolutions), ReLU.
"You will definitely want **skip connections** passing fine-grained information from the encoder
to the decoder — text strokes are thin, and without skip connections they will not survive the
bottleneck."

### `[REQ-20]` — Architecture lives in `model.py`
*Spec §3.1.* "This entire architecture will be implemented using PyTorch or TensorFlow Keras
layers in the `model.py` file."

### `[REQ-21]` — Training implemented in `train.py`, with a train/validation split
*Spec §3.2.* "The model learns directly from the training set, while the validation set is used to
evaluate its performance on unseen data at the end of each epoch."

### `[REQ-22]` — Plot training and validation loss against epochs
*Spec §3.2.* "we will plot the loss on both the training and validation sets against the number of
epochs (your synthetic test set stays untouched until Section 3.3). Analyzing this graph is
essential … **Results and plots are required for next steps.**"

### `[REQ-23]` — Address the blurriness of plain pixel-wise loss
*Spec §3.2.* "A standard pixel-wise loss like Mean Squared Error is known to produce **blurry
outputs** in image restoration — and blur is precisely the enemy when the goal is readable text."
*Hint:* "Investigate the L1 loss, the (MS-)SSIM loss, and losses computed on image *gradients*
(e.g., L1 between Sobel edge maps). Combinations of these are a well-known recipe … Text
legibility lives in the edges."
→ Combined with `[REQ-33]`, comparing loss functions is a **graded deliverable**, not optional.

### `[REQ-24]` — Evaluation implemented in `evaluate.py`, reporting PSNR and SSIM
*Spec §3.3.* Focus on "two widely-used metrics in image restoration: Peak Signal-to-Noise Ratio
(PSNR) and the Structural Similarity Index (SSIM)."

### `[REQ-25]` — Report PSNR/SSIM for training, validation and test in a single table
*Spec §3.3.* One table, three rows: Training, Validation, Test.

### `[REQ-26]` — Compute the no-model baseline first
*Spec §3.3.* "The first row is your 'do nothing' baseline — the metrics of the degraded input
itself, before any enhancement, measured on the test bucket. **Compute it first.** If your model's
scores are not clearly above this line, it is not earning its parameters."

### `[REQ-27]` — Real-photo evaluation: qualitative triplets and OCR readability
*Spec §3.3.* No clean target exists for real photos, so evaluate against the commercial baseline:
1. **Qualitative:** "rectify each photo with your annotated corners, run your model, and present
   (input, your output, reference scan) triplets. Where does your model match the app? Where does
   it fall short — and where, if anywhere, does it do better?"
2. **Readability:** "run an OCR engine on all three images — the rectified input, your enhanced
   output, and the reference scan — and compare the results, either as character error rate
   against text you transcribe for a few documents or as the engine's own confidence scores."
   Two questions: did enhancement beat the raw photo, and how close to the commercial app?

*Spec §3.3 caveat to honour in the write-up:* the reference has its own style (aggressive
contrast, whitened background, sharpening) — "'different from CamScanner' is not the same as
'worse than CamScanner.'"

### `[REQ-28]` — Discuss the synthetic-table vs real-photo relationship
*Spec §3.3.* "a model can top the synthetic test set and still fail on real photos — that gap is
the central challenge of this project."

### `[REQ-29]` — Enhancement inference pipeline
*Spec §3.4.* A function taking a rectified document image and performing: preprocess → predict →
post-process ("resize the enhanced image back to the original dimensions and convert it to a
standard 8-bit image") → visualise.

---

## D. Task 2 — Corner detection

### `[REQ-30]` — Implement **both** Approach A and Approach B, and let experiments decide
*Spec §5.* "There are two natural formulations of this problem, and — this is the interesting part
— **you will implement both and let the experiments decide which one wins.**"
- **Approach A — direct coordinate regression:** "A CNN encoder followed by fully connected layers
  that output 8 numbers: the normalized (x, y) coordinates of the four corners. Train it with an
  L1 or L2 loss on the coordinates."
- **Approach B — heatmap regression:** "Reuse your encoder-decoder machinery from Section 3 to
  predict **four heatmaps**, one per corner, each containing a Gaussian blob centered on the true
  corner location. At inference, extract the coordinates with an argmax (or a *soft-argmax* if you
  want the extraction to be differentiable). Train it with a pixel-wise loss on the heatmaps."

> ⚠️ The research report states the opposite ("strictly abandoning the direct regression
> methodology"). **The report is wrong on this point.** Both approaches are mandatory.

### `[REQ-31]` — Compare on both the synthetic test set and the real labelled photos
*Spec §5.* Metrics: "the **mean corner localization error** (average Euclidean distance between
predicted and true corners, in pixels) and a stricter success metric such as the fraction of
images where all four corners fall within a small threshold of the ground truth."
Questions to answer: "Which approach is more accurate? Which is more robust to unusual viewpoints?
Which was easier to train? **Support your verdict with numbers and failure-case
visualizations.**"

### `[REQ-32]` — Corner detection inference pipeline
*Spec §5.1.* Takes a raw document photo: preprocess (resize + normalise) → predict corners with
"your better trained model" → "map coordinates back to the original image resolution" → visualise
overlay on the original raw photo.

*Spec §5.1 hint (do this before running experiments):* "Think about *why* the two approaches might
behave differently before running the experiments, and **write your prediction down.** … Was your
prediction right?" → Record the prediction in `state/discoveries.md` **before** training.

---

## E. Degradation pipeline

### `[REQ-33]` — Implement the six specified degradation types
*Spec §4.1.* Perspective warp; scaling/resolution loss; brightness, contrast and colour cast;
illumination gradients and shadows; blur and noise; JPEG compression.

### `[REQ-34]` — Apply degradations in the specified sequence
*Spec §4.3.* For one training sample, in order:
1. Random perspective warp of the clean scan onto a random background (record the four corners).
2. Random downscale–upscale by a factor between 2 and 4.
3. Random brightness, contrast and colour-cast adjustment.
4. Multiplication by a random illumination gradient and compositing of soft shadows.
5. Gaussian blur followed by Gaussian noise.
6. JPEG re-encoding at a random quality between 30 and 80.

### `[REQ-35]` — Maintain exact input/target alignment
*Spec §4.2.* "The geometric part (the perspective warp) must be **inverted exactly** — using the
known homography — before the degraded image is paired with the clean target, while the photometric
degradations (shadows, blur, noise, compression) are applied to the input **only** and never to the
target. If the input and target drift out of alignment by even a few pixels, pixel-wise losses will
punish the model for errors it did not make."

### `[REQ-36]` — Randomise every parameter within a range
*Spec §4.4 hint.* "Randomize *every* parameter within a range rather than fixing it. A model
trained on one shadow direction learns that shadow direction, not shadows."

### `[REQ-37]` — Verify the generator
*Spec §4.4.* Visually inspect a batch: degraded input, clean target, corners overlaid on the
composited photo. "Ensure that warping the degraded photo back with the recorded homography aligns
pixel-perfectly with the target." Place generated samples next to real test photos — "if a stranger
can instantly tell which is which, your degradations are not yet realistic enough." Also: "Be
cautious of excessive degradation, which might destroy the text entirely and leave the model
nothing to recover."

---

## F. Regularisation

### `[REQ-38]` — Insert dropout into both models and retrain
*Spec §6.* "update both of your models — the enhancement network and your corner detectors — by
inserting Dropout layers and train them again to observe the difference in performance. For the
direct-regression corner detector, the fully connected layers are the classic place for Dropout;
for the encoder-decoder models, experiment with where in the architecture it helps."

### `[REQ-39]` — Report specifically whether the synthetic-to-real gap shrinks
*Spec §6.* "*In particular, does the gap between synthetic validation scores and real-photo test
scores shrink?* **Report the impact on both models.**"

---

## G. Bonus — end-to-end scanner

### `[REQ-40]` — Compose the two pipelines into an automatic scanner *(bonus)*
*Spec §7.* Take the corner pipeline, compute the homography from the four predicted corners, warp,
and feed the rectified crop into the trained enhancement network. Kornia's differentiable
`get_perspective_transform`/`warp_perspective`, `torch.nn.functional.grid_sample`, or plain
`cv2.getPerspectiveTransform`/`cv2.warpPerspective` are all acceptable.

### `[REQ-41]` — Evaluate the full chain twice *(bonus)*
*Spec §7.* "Evaluate the full chain on the real test photos and report the OCR metric and
qualitative results **twice**: once rectifying with your annotated corners, and once with predicted
corners. The difference tells you exactly how much corner errors cost the enhancement stage."

*Spec §7 hint:* "if the predicted corners are in the wrong order, the homography will flip or
rotate the page."

### `[REQ-42]` — Differentiable joint fine-tuning *(bonus, flagged Option)*
*Spec §7 🧩 Option.* "chain corner detector → warp → enhancement network and fine-tune the whole
system end-to-end with the enhancement loss. Does the corner detector improve when it is trained
for what the pipeline actually needs? Does the gap you measured above shrink?"
→ Scope decision in ADR-012: attempted only if Phases 00–08 complete cleanly.

---

## H. Submission

### `[REQ-43]` — Well-documented, modular, executable codebase
*Spec, Submission Criteria.* Must demonstrate grasp of synthetic data generation, model
architecture, loss functions, post-processing. "Be prepared to explain and modify any part of the
code if asked (e.g., adjusting hyperparameters, changing the model architecture, adding a new
degradation)."

### `[REQ-44]` — Visualise intermediate and final outputs
*Spec, Submission Criteria.* Degraded input, enhanced output, clean target on synthetic data;
reference scan on real photos; predicted corners.

### `[REQ-45]` — Include method comparisons with qualitative analysis
*Spec, Submission Criteria.* "Include comparisons between different methods (**loss functions**,
and **regression vs. heatmap**) with qualitative analysis."

### `[REQ-46]` — Provide two inference pipelines
*Spec, Submission Criteria.* One accepting an unseen rectified document image (enhancement), one
accepting an unseen raw photo (corner detection). "A single fully automatic photo-to-scan pipeline
is the bonus." Pipelines must be "robust to variations (e.g., lighting, shadows, distance,
different backgrounds)."

### `[REQ-47]` — Report the full metric set
*Spec, Submission Criteria.* PSNR and SSIM on synthetic training/validation/test splits, alongside
a no-model baseline; on real photos, OCR-based readability improvement and a qualitative comparison
against the commercial scanning app.

### `[REQ-48]` — Discuss limitations and potential improvements
*Spec, Submission Criteria.* Examples given: "curled or folded pages, extreme shadows, the
synthetic-to-real gap."

---

## I. Context requirements (affect planning, not code)

### `[REQ-49]` — The final grade is determined on a hidden test set
*Spec §1.1 🚨 and §2.3 🚨.* "the teaching staff will run your pipeline on new, unseen realistic
photos." Your own real photos are a rehearsal, not the exam.
**Planning consequence:** robustness and generalisation outrank synthetic-benchmark numbers
everywhere they conflict. See `02-research/sim2real-playbook.md`.

---

## Traceability

| Spec section | Requirements |
|---|---|
| §1.1 | REQ-01, 02, 03, 04, 49 |
| §1.2 | REQ-05 |
| §1.3 | REQ-07, 08, 09 |
| §2.1 | REQ-06 |
| §2.2 | REQ-10, 11, 12, 13 |
| §2.3 | REQ-14, 15, 16, 17, 49 |
| §2.4 | REQ-18 |
| §3.1 | REQ-19, 20 |
| §3.2 | REQ-21, 22, 23 |
| §3.3 | REQ-24, 25, 26, 27, 28 |
| §3.4 | REQ-29 |
| §4.1 | REQ-33 |
| §4.2 | REQ-35 |
| §4.3 | REQ-34 |
| §4.4 | REQ-36, 37 |
| §5 | REQ-30, 31 |
| §5.1 | REQ-32 |
| §6 | REQ-38, 39 |
| §7 | REQ-40, 41, 42 |
| Submission | REQ-43, 44, 45, 46, 47, 48 |
