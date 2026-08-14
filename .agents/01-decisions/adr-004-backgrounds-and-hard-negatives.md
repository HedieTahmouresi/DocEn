# ADR-004 — Background Sourcing, Hard Negatives, and Range Calibration

**Status:** ACCEPTED · **Date:** 2026-08-12 · **Reversibility:** High (adding data is cheap)

## Context

The spec says only "a random background image (a desk, a table, a carpet)" (§1.3). It does not say
where backgrounds come from. This turns out to be the highest-leverage under-specified decision in
the project.

Evidence: the baseline implementation dissected in the research report achieved **96% corner
success on synthetic test data and 0.00% on real photos**, mean error rising from 1.85 px to
107.44 px. The report attributes this to two coupled causes — restrictive homography sampling
(`margin_ratio=0.15`, `perspective_strength=0.08`) and insufficiently complex backgrounds. With
simple backgrounds, the highest-contrast edge in the image *is* the document, so the network learns
an edge heuristic rather than the semantic concept of a page. Put that network on a cluttered desk
and the heuristic fires on the wrong edge.

`[REQ-49]` sharpens the stakes: the grade comes from the TAs running the pipeline on **unseen**
realistic photos.

## Decision

### 1. Backgrounds: self-shot photos + DTD

**~50 self-shot phone photos** of surfaces the model will actually be tested on, plus the
**Describable Textures Dataset** (5,640 images, 47 categories, research use, from the Oxford VGG
page) for breadth.

Self-shot photos give domain match that no public texture set can: the same phone, the same rooms,
the same lighting, the same surfaces. DTD gives variety that 50 photos cannot, and prevents the
network memorising a small background set.

**Self-shot capture brief (human task):**
- Surfaces you would actually put a document on: desk, table, carpet, floor, bed, sofa, notebook.
- **At least 15 must be deliberately cluttered "hard negatives":** other sheets of paper lying
  nearby, an open book, a laptop, a keyboard, a magazine, a ruler, strong intersecting straight
  lines, a rectangular placemat. These teach the network that "a bright quadrilateral with sharp
  edges" is not sufficient evidence of *the* document.
- Vary lighting to match `[REQ-04]`: daylight, warm lamp, harsh overhead, partial shadow.
- Shoot at the same resolution range as your real test photos, no document in frame.

**Composition ratio:** roughly 50% self-shot / 50% DTD per sample, sampled at random. Log the
ratio as a config value so it can be swept if corner performance disappoints.

**Explicitly not doing:** procedural OpenCV backgrounds (wood grain, gradients, random polygons) as
the primary source. They are unconvincing and carry the highest sim2real risk. They are acceptable
only as a small additional slice if background variety turns out to be the measured bottleneck.

### 2. Hard negatives are a first-class part of the generator

Beyond cluttered background *photos*, the generator may composite **distractor quadrilaterals**
onto the background before placing the document: a second, dimmer, partially-visible page-like
rectangle. This is the explicit hard-negative-mining idea from the research report.

Treat this as `[REC]`, not `[REQ]`: implement the plain generator first, establish a baseline, and
add distractors as a **measured ablation** in Phase 06. If corner accuracy on real cluttered photos
is already good, distractors add complexity for nothing — that is the scope-guard call.

### 3. Calibrate parameter ranges against measured real-photo statistics — then widen

This is the strongest available defence against the sim2real gap, and the spec sanctions it
directly: §1.1 hint says "whatever degradations you see — shadows, blur, colour casts, perspective
distortion — are exactly what your synthetic pipeline in Section 4 must reproduce", and §4.4 asks
you to place generated samples beside real photos and check that a stranger cannot tell them apart.

**Procedure (Phase 01 measures, Phase 02 consumes):**
1. From the annotated real photos, measure the distribution of: page area as a fraction of frame;
   in-plane rotation; perspective severity (deviation of the corner quad from a rectangle, e.g.
   the ratio of opposite side lengths); margin from frame edge; mean brightness and contrast;
   estimated blur (variance of Laplacian); presence and direction of shadow.
2. Write these to a **real-photo degradation profile** consumed by the generator config.
3. **Set generator ranges to cover the observed distribution and then widen by a healthy margin —
   roughly 1.5–2× the observed spread.**

The widening is the important half. Fitting the generator tightly to your own 10–15 photos is
overfitting to a 15-sample estimate of reality, and the graded set is a *different* set of photos
(`[REQ-49]`). Domain-randomisation results consistently favour ranges wider than the target
domain: the model should see the real distribution as an easy interior case, not as the boundary.

**Honesty note.** Measuring statistics of the test photos to configure training is a mild form of
information leakage. It is sanctioned by the spec, it uses aggregate statistics rather than
per-image labels, and the deliberate widening blunts it. **Disclose it in the report** — it is a
thoughtful methodological point, not something to hide.

### 4. Kill the baseline's spatial priors explicitly

Directly countering the diagnosed failure:
- Page area varies widely (target ~15%–95% of the frame), not a fixed margin.
- Margin from the frame edge can approach zero — corners may sit right against the border.
- Perspective severity spans from near-fronto-parallel to strong, well past `strength=0.08`.
- Every parameter is redrawn per sample (`[REQ-36]`).
- **All four corners stay inside the frame.** Truncated pages are out of scope: a corner outside
  the frame has no heatmap peak, and real scanner use has the whole page in view. Note this as a
  limitation under `[REQ-48]` rather than engineering for it.

Concrete ranges live in `03-spec/synthetic-generator-spec.md`.

## Consequences

**Good.** Attacks the documented failure mode directly. Domain-matched backgrounds cost ~15 minutes
of human time. Range calibration is measurable and reviewable, not guesswork.

**Costs.** A human dependency in Phase 01 (photo capture) that gates Phase 02's calibration —
though Phase 02 can begin with provisional ranges. DTD is a ~600 MB download. Storage and sampling
logic for two background pools.

**Risk.** Backgrounds may still be insufficiently diverse. Detection: if corner performance is
strong on synthetic and weak on real *cluttered* photos specifically, backgrounds are the suspect —
before touching the architecture, add background variety and distractors. `[REC]`

## Alternatives considered

- **DTD only.** Zero human effort, good texture variety, but flat close-up textures with no
  clutter, no depth, no competing document edges — weakest exactly where it matters.
- **Self-shot only.** Perfect domain match, but ~50 surfaces is few enough that the network can
  memorise them.
- **Procedural only.** Unlimited and fully OpenCV-native, but visually unconvincing; highest risk.
- **Public document-in-the-wild datasets (e.g. SmartDoc) as backgrounds.** Rejected: they contain
  documents, which would inject unlabelled positives into the background pool.
