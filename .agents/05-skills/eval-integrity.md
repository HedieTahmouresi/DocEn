# Skill: Evaluation Integrity

**Load before:** computing or reporting any number that goes in the report. **Mandatory in
Phases 05, 06, 07, 08.**

This is about not fooling yourself. Every item here is a way to produce a number that is technically
computed correctly and substantively misleading — and most of them are easy to do by accident and
hard to spot afterwards.

---

## 1. The test set is touched once

`[CON-07]`, spec §2.3: held out and "touched once, at the end, to report final numbers". Spec §3.2:
"your synthetic test set stays untouched until Section 3.3."

**Forbidden:**
- Selecting the best epoch on test
- Choosing the winning loss variant on test
- "Just checking" test mid-project
- Re-running test after a tweak

**Why it matters beyond compliance:** each peek makes the test number a little more optimistic, and
the whole point of that number is that it is the honest one. Validation exists for every decision;
test exists for one report.

**If you accidentally look:** say so in the report. A disclosed peek costs a caveat; an undisclosed
one is a misreported result.

---

## 2. Compare like with like

The most common source of a wrong-but-plausible number.

| Trap | Fix |
|---|---|
| Different `frozen_version` between compared runs | Bump and regenerate deliberately; never mix in one table |
| Different resolution | Everything is 512 (ADR-002) — verify, don't assume |
| Different batch size (changes BatchNorm statistics) | Hold it constant across an ablation |
| OCR at each image's natural resolution | The matched-resolution protocol (ADR-011 §5) |
| Different Tesseract version across machines | Run all OCR on **one** machine |
| Corner error in px without stating the resolution | Report px @512 **and** % of diagonal |
| Success rate without its threshold | State it in the same sentence |
| PSNR with the wrong `data_range` | 1.0 on `[0,1]` tensors — a 255 shifts everything ~48 dB |
| SSIM with mismatched window parameters | Pin them (ADR-010); report them |

---

## 3. Metric mechanics that change the answer

**PSNR must be per image, then averaged.** Because of the logarithm, the mean of per-image PSNRs is
not the PSNR of the pooled MSE. Only the first is comparable to anything published.

**SSIM defaults differ between implementations.** `skimage` defaults to a uniform 7×7 window; the
conventional SSIM uses an 11×11 Gaussian with σ=1.5. Report which you used — an SSIM without its
settings is not reproducible.

**The no-model baseline is a bug detector, not just a requirement.** `[REQ-26]` says compute it
first. If the baseline *beats* the model, something is inverted, misaligned, or in the wrong colour
space — you have found a bug before writing the report.

---

## 4. The real photos are the measurement instrument

`[CON-06]`: never train on them, never validate on them, never run the degradation pipeline on
them.

**The grey area, handled explicitly.** ADR-004 permits measuring their *aggregate statistics* to
calibrate the generator, and the spec sanctions it (§1.1: "whatever degradations you see … are
exactly what your synthetic pipeline in Section 4 must reproduce"). Three conditions make it sound:

1. Aggregate statistics only — never per-image labels
2. Ranges deliberately **widened** beyond the observed spread (ADR-004 §3)
3. **Disclosed in the report**

The reason the widening matters: the grade comes from a *hidden* set of photos (`[REQ-49]`). Fitting
the generator tightly to your own 20 photos is overfitting to a 20-sample estimate of reality.

**Also avoid the softer leak:** iterating on the model until the real-photo numbers improve turns
the real set into a validation set. If you do it, say so — and note that the hidden test set is
where the real answer lies.

---

## 5. Report what you found, not what you hoped

Concrete cases in this project where the uncomfortable result is the correct thing to report:

| Situation | Do |
|---|---|
| Full-resolution raw input OCRs better than the model output | **Report it.** It is a real limitation of the 512 resolution choice — `[REQ-48]` |
| The MSE model wins on PSNR while looking worse | **Report and explain it.** PSNR is a monotone function of MSE, so L-A is directly optimising it — this is the insight `[REQ-45]` rewards |
| The Sobel term did nothing | **Report it.** A null result is a result |
| The dropout gap did not shrink | **Report it**, and argue what it implies: the gap is distribution mismatch, not memorisation |
| Approach A failed badly | **Report it** — after confirming it is not a bug (`training-diagnostics.md`) |
| Tesseract confidence fell while CER improved | **Report both**, with the documented miscalibration caveat |
| The model is worse than CamScanner | **Report it.** Spec §3.3 supplies the fair framing: "different from CamScanner is not the same as worse than CamScanner" |

**Never revise a protocol after seeing the result it produced.** If a protocol is wrong, it was
wrong before you ran it, and the fix must be justifiable without reference to the outcome.

---

## 6. Reproducibility

- Every number lives in a `metrics.json`, in a run directory, with the git commit and
  `frozen_version`.
- `evaluate.py` regenerates it from a checkpoint.
- Fix the seed and record it — but understand that what actually guarantees comparability here is
  the **frozen sets**, not the seed.
- **Do not report the best of several seeds.** If you ran several, report the spread or say which
  you used and why.

---

## 7. Statistical honesty with small samples

Real numbers in this project come from 15–25 photos and 5 transcribed documents. That is small.

- Report per-document CER as well as the mean — one bad document dominates an average of five.
- Do not claim a difference is meaningful when it rests on one or two photos.
- With ~500 frozen test samples, synthetic metrics are reasonably stable; with 20 real photos they
  are not. **Say so.** Acknowledging the sample size is a strength, not a weakness.
- Prefer showing the distribution (per-photo scatter) over a single mean where you can.

---

## Pre-report checklist

```
[ ] Test split touched exactly once; no test-based selection anywhere
[ ] Every compared run shares frozen_version, resolution, batch size
[ ] PSNR per image then averaged; data_range = 1.0
[ ] SSIM parameters pinned and reported
[ ] No-model baseline computed first, and the model beats it
[ ] OCR under the matched-resolution protocol; all OCR on one machine
[ ] Full-resolution raw row reported separately
[ ] Corner errors carry resolution and threshold
[ ] Real photos never trained on; calibration use disclosed and widened
[ ] Uncomfortable results reported, not omitted
[ ] Every number traces to a metrics.json
[ ] Small-sample caveats stated
[ ] No protocol was revised after seeing its result
```
