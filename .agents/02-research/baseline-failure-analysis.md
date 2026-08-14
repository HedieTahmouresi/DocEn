# Case Study: The 96% → 0% Collapse

The most useful piece of evidence available for this project is a **failure**. A prior peer
implementation, analysed in `Document Scanner Implementation Plan.md`, produced a corner detector
that looked excellent and was worthless.

Read this before writing the generator. It is the single most instructive artifact in the source
material, and the entire sim2real strategy is a response to it.

---

## The numbers

| Model | Synthetic test | Real photos |
|---|---|---|
| **Approach A** (coordinate regression) | 8.00% success · 10.41 px mean error | — |
| **Approach B** (heatmap regression) | **96.00% success · 1.85 px mean error** | **0.00% success · 107.44 px mean error** |

Approach B did not degrade on real photos. It **inverted**. Mean error rose by a factor of ~58, and
not a single real photo had all four corners localised.

> ⚠️ **Provenance.** These figures are reported by the research report, citing a baseline notebook
> (`cv_project_notebook (78).ipynb`) that **we do not have**. Treat them as *reported*, not
> verified. The resolution and success threshold behind them are undocumented, so the absolute
> magnitudes are not comparable to ours. **The pattern is the transferable finding, not the
> numbers** — do not benchmark against 1.85 px or 107.44 px.

---

## What actually happened

A 96% synthetic success rate is not evidence that the network learned what a document corner is. It
is evidence that the network learned **the generator**.

The report identifies the mechanism in the baseline's data-generation parameters:

```
margin_ratio         = 0.15      # document always inset 15% from the frame edge
perspective_strength = 0.08      # very mild perspective distortion
```

Both are **fixed**, not sampled. The consequences compound:

1. **The document's position was nearly constant.** Every training image had the page in
   approximately the same place at approximately the same scale. The optimal strategy for a network
   is then to output that position — with a small correction from image evidence. Under this
   distribution, that strategy scores 96%.

2. **Perspective was nearly absent.** At `strength=0.08` the quadrilateral is close to a rectangle.
   The network never learned that a document corner can be acute, or that opposite edges can
   converge steeply.

3. **The prior was learned instead of the concept.** Show that network a page at 40° filling 90% of
   a cluttered frame and it has nothing to fall back on. Hence 107 px — roughly the distance from
   the memorised position to the actual one.

4. **Backgrounds compounded it.** The report notes the backgrounds likely lacked structural
   complexity. On a plain background the document *is* the highest-contrast quadrilateral, so a
   contrast-edge heuristic suffices. On a real desk with other papers, that heuristic fires on the
   wrong object.

The deeper point: **a synthetic validation set drawn from the same generator cannot detect this.**
Val and test were sampled from the same narrow distribution, so they agreed with training. The only
signal that anything was wrong came from the real photos — which is exactly why `[REQ-16]` makes
them a separate, mandatory fourth evaluation set.

---

## What this project does differently

Each countermeasure and where it is specified:

| Failure cause | Countermeasure | Where |
|---|---|---|
| Fixed margin ratio | Page area sampled ~15%–95% of frame; margins may approach zero | ADR-004 §4, `03-spec/synthetic-generator-spec.md` |
| Fixed perspective strength | Perspective severity sampled across a wide range, well past 0.08 | same |
| Any fixed parameter | `[REQ-36]`: every parameter randomised per sample | spec §4.4 |
| Simple backgrounds | ~50 self-shot photos incl. ≥15 cluttered hard negatives, plus DTD | ADR-004 §1 |
| No competing structure | Optional distractor quadrilaterals, as a measured ablation | ADR-004 §2 |
| Ranges guessed | Ranges calibrated to measured real-photo statistics, **then widened ~1.5–2×** | ADR-004 §3 |
| Blind to the gap | Real-photo metrics reported beside synthetic ones, always as a pair | ADR-011 §4 |

---

## The lesson to carry into every phase

> **A synthetic score is a measurement of your generator, not of your model.**

Practical consequences for how you work:

- **Never celebrate a synthetic number alone.** Report it next to the real-photo number from the
  first moment both exist. A widening gap is the alarm.
- **Treat a very high synthetic score as suspicious**, not as success. 96% on synthetic corner
  detection was the *symptom*. If Approach B exceeds ~95% synthetic success early, check the
  generator's parameter ranges before believing it.
- **Diversity of the generator beats capacity of the model.** Reaching for a bigger network when
  real-world performance is poor is almost always the wrong move here. Widen the data distribution
  first.
- **This is also why dropout gets its own phase.** `[REQ-39]` asks specifically whether the
  synthetic-to-real gap shrinks — the spec is pointing at exactly this failure mode.

---

## What the baseline got right

Worth keeping, per the report:

- **Heatmap regression over coordinate regression** was the correct architectural instinct — 96% vs
  8% on the same synthetic data is a large, real signal about the representation, even if the 96%
  was inflated by a narrow distribution.
- **OCR-based end-to-end evaluation.** The baseline measured downstream OCR confidence, tying
  geometric accuracy to actual readability rather than stopping at pixel distance. The spec requires
  this too (`[REQ-27]`), and ADR-011 refines it into a resolution-fair protocol.

---

## A diagnostic you can run

If Phase 06 produces strong synthetic and weak real corner performance, this test localises the
cause before you touch the architecture:

1. Take the real photos. Measure their page-area fraction, in-plane rotation, and perspective
   severity (ADR-004 §3).
2. Plot those against the **same statistics sampled from your generator**.
3. **If the real distribution sits outside or at the edge of the synthetic one, the generator is the
   problem** — not the network, not the loss, not the capacity.

That plot is worth including in the report regardless of the outcome. It is direct evidence of
whether the sim2real gap was addressed, and it is the kind of analysis `[REQ-28]` asks for.
