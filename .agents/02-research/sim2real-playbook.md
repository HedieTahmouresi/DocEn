# The Sim2Real Playbook

The central technical problem of this project, and the thing to optimise for above all synthetic
benchmarks.

**Why it dominates everything else:** `[REQ-49]` — the grade is assigned by the teaching staff
running your pipeline on **new, unseen realistic photos** at the presentation. Not on your synthetic
test set. Not even on your own real photos. A model that tops the synthetic table and fails on a
TA's photo of a page on a cluttered desk scores badly, and `02-research/baseline-failure-analysis.md`
shows that outcome is the *default*, not an unlucky edge case.

---

## The one-sentence strategy

> Make the real world an **easy interior sample** of your training distribution — not a point on
> its boundary, and never outside it.

Everything below follows from that sentence.

---

## 1. Domain randomisation, and why ranges should be *wider* than reality

The instinct when calibrating a synthetic generator is to match reality: measure how much blur real
photos have, and generate that much blur. This is subtly wrong.

Domain randomisation works by making the model treat real data as **just another variant** of a
distribution it has already mastered. If the synthetic range exactly matches the observed real
range, then roughly half of real samples fall in the harder half of the distribution, and the
tails — which is where failures live — are unrepresented. Widen the range and the entire real
distribution sits comfortably inside the region the model handles well.

The literature consistently supports this: increasing synthetic diversity, especially varied
viewpoints and complex backgrounds, is what bridges the gap.

**Concretely (ADR-004 §3):** measure the real photos' statistics, then set generator ranges to
roughly **1.5–2× the observed spread**, centred on the observed centre.

**The counter-pressure — `[REQ-37]`:** "Be cautious of excessive degradation, which might destroy
the text entirely and leave the model nothing to recover." Widening has a limit. The test is
whether the *hardest* samples your generator produces are still plausibly recoverable by a human
eye. If you cannot read the text, the network is being trained on an impossible task and will learn
to hallucinate.

**Calibration procedure:**
1. Generate 100 samples at the proposed ranges.
2. Sort by severity. Look at the worst 10.
3. If any are unreadable → the upper bound is too high.
4. If none are noticeably harder than your hardest real photo → too low.

---

## 2. The parameter-coverage plot

The single most useful diagnostic in this project, and it takes an hour to build.

For each of these, plot the **distribution from your generator** against the **measured values from
your real photos**:

| Property | How to measure on a real photo |
|---|---|
| Page area fraction | area of the annotated quad / image area |
| In-plane rotation | angle of the TL→TR edge |
| Perspective severity | ratio of opposite side lengths of the quad; deviation from a parallelogram |
| Margin from frame edge | min distance from any corner to the border, normalised |
| Mean brightness / contrast | over the annotated page region only |
| Blur | variance of the Laplacian over the page region |
| Colour cast | per-channel mean ratios within the page region |

**Reading it:**
- Real distribution well inside synthetic → good.
- Real distribution at the edge → the model is extrapolating. Widen.
- Real distribution **outside** synthetic → this is your bug. Found before wasting a training run.

Put this plot in the report. It is direct evidence of how the sim2real gap was addressed and speaks
to `[REQ-28]` and `[REQ-48]`.

---

## 3. Backgrounds are the highest-leverage variable

Full decision in ADR-004. The reasoning, briefly:

A corner detector can succeed on simple backgrounds using a shallow heuristic — *find the
highest-contrast quadrilateral*. That heuristic gets near-perfect synthetic scores and collapses on
a desk with other papers on it. The only way to prevent the network learning it is to make it
**insufficient during training**: put competing quadrilaterals, strong straight lines and other
documents in the background, so contrast alone cannot identify the target.

This is hard negative mining applied to synthesis. It is the difference between learning "the
brightest rectangle" and learning "the page".

**Priority order if corner performance on real photos disappoints:**
1. Background diversity and clutter ← start here
2. Homography range width
3. Photometric range width
4. Heatmap σ / loss
5. Architecture ← almost never the answer

---

## 4. The six degradations and what each defends against

`[REQ-33]`/`[REQ-34]` fix the list and the order. Understanding *why* each is there tells you what
you break if you weaken one.

| Degradation | Real-world cause | Defends against |
|---|---|---|
| Perspective warp | Photo taken at an angle | The whole point of corner detection; produces the labels for free |
| Downscale–upscale ×2–4 | Photographed from a distance; limited sensor resolution | Model expecting crisp input; forces genuine detail reconstruction |
| Brightness / contrast / colour cast | Incandescent vs LED vs daylight | Model assuming a fixed white point |
| **Illumination gradient + soft shadows** | Uneven lighting; hand or phone blocking overhead light | **The characteristic defect of phone document photos** — and the one a scanner must fix |
| Gaussian blur + noise | Camera shake, imperfect focus, sensor noise | Model relying on sharp edges that real photos do not have |
| JPEG re-encode q30–80 | Phones store compressed | Model unprepared for block artifacts and colour subsampling |

The illumination row is the one that matters most for enhancement. It is also the one where
"randomise every parameter" bites hardest: a model trained on a single shadow direction learns that
direction, not shadows (`[REQ-36]`). Randomise gradient direction, gradient steepness, shadow
polygon shape, position, softness, opacity, and **whether a shadow is present at all**.

---

## 5. Two systematic gaps you cannot close, and must instead disclose

Honest limitations for `[REQ-48]`:

**Paper is not flat.** The synthetic pipeline applies a *homography*, which assumes a planar page.
Real pages curl, fold and bend. No amount of parameter widening produces a curled page from a
homography. Curl is explicitly out of scope (`00-project/project-brief.md`); the fix would be a thin
plate spline or a learned dewarping field. **Say this in the limitations.**

**Synthetic shadows are geometric, real shadows are physical.** Blurred polygons approximate soft
shadows but do not model penumbra falloff, coloured bounce light, or the interaction of shadow with
paper texture. Expect residual error on the hardest real lighting.

---

## 6. The "stranger test" — `[REQ-37]`

The spec provides its own acceptance test, and it is a good one:

> "place a few generated samples next to the real test photos — if a stranger can instantly tell
> which is which, your degradations are not yet realistic enough."

Make this a formal Phase 02 gate item. Build a figure with a shuffled grid of synthetic and real
samples and look at it honestly. Common giveaways to check for specifically:

- Synthetic backgrounds are too *flat* — real surfaces have depth-of-field falloff and specular
  variation.
- Shadow edges are too *geometric* — real penumbras are wider and irregular.
- The document is too *evenly* lit within itself — real pages have subtle gradients even in good light.
- The composite has a visible seam at the page boundary — real pages have a thin shadow along their
  edge where they meet the surface. **Adding a subtle edge shadow is a cheap, high-value realism
  win** and stays within `[CON-03]` (it is an OpenCV blur and blend).
- Noise looks uniform — real sensor noise is stronger in shadows than in highlights.

---

## 7. Priorities when the gap is measured

Once you have both synthetic and real numbers, the gap tells you where to spend effort:

| Symptom | Most likely cause | First action |
|---|---|---|
| Synthetic ≫ real, corners | Generator too narrow / backgrounds too simple | Coverage plot (§2), then widen and add clutter |
| Synthetic ≫ real, enhancement | Degradations unrealistic — probably illumination | Stranger test (§6); compare real vs synthetic shadow statistics |
| Both poor | Under-training or a bug, not a domain gap | `05-skills/training-diagnostics.md` |
| Real good, synthetic poor | Almost certainly an evaluation bug | Check the frozen set, normalisation, corner ordering |
| Gap shrinks with dropout | Genuine over-fitting to synthetic artifacts | Report it — this is exactly `[REQ-39]` |
