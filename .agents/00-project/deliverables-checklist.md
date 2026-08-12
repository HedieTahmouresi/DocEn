# Deliverables Checklist

Grading criteria from the spec's Submission Criteria section, mapped to the concrete artifact that
satisfies each, and to the phase that produces it.

**How to use:** this is the definition of "the project is finished". Re-read it at the start of
Phase 10, and spot-check it at every phase gate — it is easy to complete all the phases and still
be missing a required *artifact* (a plot, a table row, a discussion paragraph).

Status column: `[ ]` not started · `[~]` partial · `[x]` complete and verified.

---

## 1. Code implementation & explanation

*Spec: "well-documented, modular, and executable codebase"; "demonstrate a strong grasp of all
concepts"; "be prepared to explain and modify any part of the code if asked".*

| ☐ | Artifact | Where | Req | Phase |
|---|---|---|---|---|
| [ ] | `model.py` containing all three architectures, hand-written | repo root or `src/` | REQ-20 | 04, 06 |
| [ ] | `train.py` — configurable training entry point | | REQ-21 | 04 |
| [ ] | `evaluate.py` — all reported metrics | | REQ-24 | 05 |
| [ ] | Synthetic generator, OpenCV+NumPy only | `src/data/` | REQ-07, CON-03 | 02 |
| [ ] | Dataset/DataLoader with on-the-fly generation | `src/data/` | REQ-11 | 03 |
| [ ] | Two inference pipelines (see §3 below) | `src/pipeline/` | REQ-29, 32 | 05, 06 |
| [ ] | Config-driven (no magic constants) — supports live hyperparameter changes at presentation | `configs/` | REQ-43 | 00 |
| [ ] | README explaining how to run each entry point | repo root | REQ-43 | 10 |
| [ ] | Own SSIM/MS-SSIM implementation + numerical validation test | `src/losses/`, `tests/` | ADR-010 | 04 |

**Presentation-readiness check.** For each of these, can you explain it *without reading it*:
why skip connections, why L1 over L2, why heatmaps beat regression (or didn't), what each of the
six degradations simulates, why the val/test sets are frozen, why BatchNorm is not "explicit
regularisation"? If not, that is a gap to close before Phase 10 ends.

---

## 2. Visualisation of results

*Spec: "visualize intermediate and final outputs"; "include comparisons between different methods
(loss functions, and regression vs. heatmap) with qualitative analysis".*

| ☐ | Figure | Contents | Req | Phase |
|---|---|---|---|---|
| [ ] | Generator sanity panel | degraded input · clean target · corners overlaid on composite | REQ-18, 37 | 02 |
| [ ] | Round-trip alignment proof | photometrics-off round-trip vs target, with PSNR annotated | REQ-35, 37 | 02 |
| [ ] | Synthetic vs real side-by-side | the "can a stranger tell which is which" test | REQ-37 | 02 |
| [ ] | Loss curves | train + val vs epoch, **one plot per model** | REQ-22 | 04, 06 |
| [ ] | Enhancement results (synthetic) | degraded · model output · clean target, several samples | REQ-44 | 05 |
| [ ] | Enhancement results (real) | **triplets:** rectified input · your output · reference scan | REQ-27, 44 | 05 |
| [ ] | Loss-function comparison | same input, output from each of the 4 loss variants, side by side | REQ-45 | 05 |
| [ ] | Predicted corners overlay | on raw real photos, GT vs predicted, colour-coded per §8 of conventions | REQ-32, 44 | 06 |
| [ ] | Corner failure cases | worst predictions from **both** approaches, with commentary | REQ-31 | 06 |
| [ ] | Heatmap visualisation | the four predicted heatmaps for a sample | REQ-44 | 06 |
| [ ] | Dropout comparison | before/after, both models | REQ-38 | 07 |
| [ ] | End-to-end results | full chain output on real photos | REQ-40 | 08 |
| [ ] | Annotated-vs-predicted corners | the same photo rectified both ways, side by side | REQ-41 | 08 |

---

## 3. Pipeline for unseen data

*Spec: "provide two inference pipelines: one accepting an unseen rectified document image
(enhancement), one accepting an unseen raw photo (corner detection). A single fully automatic
photo-to-scan pipeline is the bonus." Must be "robust to variations (e.g., lighting, shadows,
distance, different backgrounds)."*

| ☐ | Pipeline | Steps required by spec | Req | Phase |
|---|---|---|---|---|
| [ ] | **Enhancement** | preprocess → predict → resize back to original dims + 8-bit → visualise | REQ-29 | 05 |
| [ ] | **Corner detection** | preprocess → predict (better model) → map coords to original res → overlay | REQ-32 | 06 |
| [ ] | **Bonus: full scanner** | raw photo → corners → homography → warp → enhance → clean scan | REQ-40 | 08 |

**Robustness gate.** `[REQ-49]` means the TAs will run these on photos you have never seen. Before
Phase 10, run all three pipelines on the *hardest* photos you own and confirm they do not crash,
do not produce a flipped page, and degrade gracefully. A pipeline that throws on an unusual aspect
ratio loses more marks than one that returns a mediocre result.

---

## 4. Performance metrics & analysis

*Spec: "report PSNR and SSIM on the synthetic training, validation, and test splits, alongside a
no-model baseline; on real photos, report OCR-based readability improvement and a qualitative
comparison against the commercial scanning app."*

### 4a. Enhancement — the required table

| ☐ | Row | Note | Req |
|---|---|---|---|
| [ ] | **No-model baseline** (degraded input vs target, test bucket) | **compute this first** | REQ-26 |
| [ ] | Training | | REQ-25 |
| [ ] | Validation | | REQ-25 |
| [ ] | Test | the honest headline number | REQ-25 |

Plus the interpretation the spec asks for: large train-vs-test gap ⇒ overfitting; small gap with
poor numbers everywhere ⇒ underfitting. Say which yours is.

### 4b. Enhancement — real photos

| ☐ | Item | Req |
|---|---|---|
| [ ] | OCR on all three (rectified input, your output, reference scan) at matched resolution | REQ-27 |
| [ ] | CER against hand-transcribed text for a few documents **and/or** engine confidence | REQ-27 |
| [ ] | Did enhancement beat the raw photo? By how much? | REQ-27 |
| [ ] | How close to the commercial app? | REQ-27 |
| [ ] | Fairness caveat acknowledged (different ≠ worse) | REQ-27 |

### 4c. Corner detection

| ☐ | Item | Req |
|---|---|---|
| [ ] | Mean corner localisation error (px @512 **and** % of diagonal), Approach A & B | REQ-31 |
| [ ] | Success rate (all 4 corners within threshold), both approaches, threshold stated | REQ-31 |
| [ ] | Both measured on synthetic test **and** on real labelled photos | REQ-31 |
| [ ] | Written verdict: which is more accurate / more robust / easier to train | REQ-31 |
| [ ] | Pre-registered prediction recorded *before* experiments, and its outcome | spec §5.1 hint |

### 4d. Dropout ablation

| ☐ | Item | Req |
|---|---|---|
| [ ] | Enhancement: metrics with and without dropout | REQ-38 |
| [ ] | Corner detectors: metrics with and without dropout | REQ-38 |
| [ ] | **Explicit answer:** did the synthetic-val → real-test gap shrink? | REQ-39 |

### 4e. Bonus

| ☐ | Item | Req |
|---|---|---|
| [ ] | Full chain evaluated on real photos with **annotated** corners | REQ-41 |
| [ ] | Full chain evaluated on real photos with **predicted** corners | REQ-41 |
| [ ] | The difference, interpreted: what do corner errors cost? | REQ-41 |
| [ ] | *(Optional)* joint fine-tuning result and whether the gap shrank | REQ-42 |

### 4f. Discussion

| ☐ | Item | Req |
|---|---|---|
| [ ] | Limitations discussed | REQ-48 |
| [ ] | — curled / folded pages | REQ-48 |
| [ ] | — extreme shadows | REQ-48 |
| [ ] | — the synthetic-to-real gap | REQ-48, 28 |
| [ ] | — the resolution ceiling on OCR (see `02-research/evaluation-and-ocr.md`) | REQ-48 |
| [ ] | Potential improvements proposed | REQ-48 |
| [ ] | Relationship between synthetic table and real-photo performance | REQ-28 |

---

## 5. Administrative

| ☐ | Item | Req | Owner |
|---|---|---|---|
| [ ] | 10–15 real photos captured, diverse per REQ-04 | REQ-02, 04 | **human** |
| [ ] | Commercial reference scan for each photo | REQ-03 | **human** |
| [ ] | Corners annotated in RoboFlow, consistent order | REQ-05 | **human** |
| [ ] | RoboFlow project link uploaded to Google Sheets, TA-accessible | REQ-06 | **human** |

These four are human actions. The agent cannot complete them, and Phases 01+ depend on them —
surface them early and keep them in `state/STATUS.md` until done.

---

## Final sweep before submission

- [ ] Every `[REQ]` in `requirements.md` is either satisfied or has an approved entry in
      `state/deviations.md`.
- [ ] Every `[CON]` in `constraints.md` verified against the actual final code, not from memory.
- [ ] Every number in the report traces to a `metrics.json` in a run directory.
- [ ] Every run in the report appears in `state/experiments.md`.
- [ ] The repo runs end-to-end from a clean clone, following only the README.
- [ ] Negative results and abandoned approaches are documented (they earn marks under
      "demonstrating deep understanding" and cost nothing).
