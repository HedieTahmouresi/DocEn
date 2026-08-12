# Phase 01 — Real Test Set & Annotation

## Objective

Build the only honest measurement instrument in the project: real smartphone photos of documents the
model has never seen, with verified corner annotations and a commercial reference scan each.

This set does two jobs. It is the evaluation set for every real-world number in the report — and it
supplies the **statistics that calibrate the synthetic generator** (ADR-004 §3). Because of the
second job it sits on the critical path for Phase 02, not just for Phase 05.

**Start this on day one of the project.** Most of the work is human and has days of latency.

## Prerequisites

Phase 00 capture briefs delivered. Not blocked by the provided scans.

## Requirements in force

`[REQ-02]` 10–15 photos of unseen documents · `[REQ-03]` reference scan each ·
`[REQ-04]` diversity · `[REQ-05]` corner annotation, consistent order · `[REQ-06]` share TA link ·
`[REQ-10]` parse COCO keypoints · `[REQ-16]` fourth evaluation set ·
`[CON-06]` never train on these · ADR-004 §3 calibration

---

## Tasks

### A. Capture — **human**
1. **20–25 photos** (spec requires 10–15; more is strictly better and costs minutes — this is your
   only preview of reality, and it also feeds the calibration).
2. Documents **never seen in any form** in the scan set (`[REQ-02]`). Verify by eye.
3. Diversity per `[REQ-04]`, deliberately not accidentally:
   - *Lighting:* daylight · warm indoor lamp · harsh overhead · **a shadow falling across the page**
   - *Viewpoint:* angles, distances, rotations
   - *Background:* desk · carpet · **cluttered table with other papers**
   - *Camera:* slight shake, imperfect focus
   - *Document:* dense text · sparse text · a figure · a coloured logo
4. **Produce the reference scan at capture time**, while the document is still in front of you
   (spec §1.1 hint) — reconstructing them later is tedious. CamScanner, Adobe Scan, or the phone's
   built-in scanner.
5. Filenames must correspond exactly between `real/photos/` and `real/reference/`.

### B. Annotation — **human**
6. RoboFlow keypoint mode (CVAT or Label Studio acceptable). Four keypoints per photo.
7. **Order: TL, TR, BR, BL** — top-left of the *page*, not of the image
   (`00-project/conventions.md` §1). For a rotated page, TL is the page's own top-left.
8. Export **COCO keypoint JSON** into `$DATA_ROOT/real/annotations/`.
9. **`[REQ-06]`** — upload the RoboFlow project link to the Google Sheet; confirm it is public or
   TA-accessible.

### C. Parse and verify — agent
10. `src/data/annotations.py`: parse the COCO export into ordered `(4,2) float32` arrays
    (`[REQ-10]`).
11. **Render every annotation** with the colour code (`conventions.md` §8) and inspect **all** of
    them. Ordering errors are invisible in JSON and fatal downstream.
12. Assert: 4 keypoints per image, coordinates in bounds, quad convex, ordering correct by
    cross-product sign.
13. **Check EXIF orientation.** If a photo displays upright but loads sideways in `cv2.imread`, the
    annotations will not match the loaded array. Resolve now, once, in `src/utils/io.py`.

### D. Transcription — **human**
14. Hand-transcribe the text of **5 documents** for CER (ADR-011 §6) into
    `$DATA_ROOT/real/transcripts/`. Choose documents with genuinely readable text — transcribing a
    figure-heavy page is wasted effort.

### E. Calibration profile — agent, feeds Phase 02
15. Measure across the annotated photos (ADR-004 §3, `sim2real-playbook.md` §2):
    page area fraction · in-plane rotation · perspective severity · margin from frame edge ·
    mean brightness and contrast within the page · blur (variance of Laplacian) · colour cast.
16. Write a **real-photo degradation profile** the generator config consumes.
17. **Set generator ranges to cover the observed distribution, then widen ~1.5–2×.** The widening is
    the important half — see ADR-004 §3 for why fitting tightly to 20 photos is overfitting.
18. Note in the report that this calibration was done and why it is methodologically sound
    (spec §1.1 and §4.4 sanction it explicitly).

### F. Prepare the evaluation bucket
19. `RealPhotoDataset` per `[REQ-16]`: rectified-with-annotated-corners + resized reference for
    enhancement; raw resized with scaled corners for corner detection.
20. **Assert in code** that the degradation pipeline can never touch this bucket (`[CON-06]`).

---

## Gate

- [ ] ≥15 photos captured (target 20–25), each with a matching reference scan by filename
- [ ] No photographed document appears in the scan set
- [ ] Diversity audited against `[REQ-04]` — each category actually represented, listed explicitly
- [ ] All photos annotated; COCO JSON parses cleanly
- [ ] **Every annotation visually verified** for TL/TR/BR/BL ordering — all of them, not a sample
- [ ] All quads convex, in bounds, correctly ordered by the sign test
- [ ] EXIF orientation handled and tested
- [ ] `[REQ-06]` RoboFlow link uploaded and TA-accessible
- [ ] 5 documents transcribed
- [ ] Calibration profile computed and written
- [ ] Generator ranges updated from it, **widened**, and the widening factor recorded
- [ ] `RealPhotoDataset` loads; degradation-pipeline isolation asserted in code

---

## Failure modes

**Inconsistent corner ordering.** The highest-risk item in the phase. It does not crash — it
produces a homography that flips or rotates the page, and the failure surfaces much later looking
like a model problem. Spec §1.2 calls it out: labels that mix up TL and BR "silently break" both
evaluation and rectification. **Verify every single annotation visually.**

**"Top-left" defined in image space.** For a page rotated 40°, the page's TL is not the corner
nearest the image's top-left. Getting this wrong on rotated photos gives an inconsistent convention
across the set — the worst case, because most images look fine.

**Photos too easy.** If every photo is well-lit and near-overhead, the real test set will not reveal
the sim2real gap, and the TAs' harder photos will. Deliberately include the hard cases.

**Documents that overlap the scan set.** Violates `[REQ-02]` and invalidates the generalisation
claim, which is the entire point of this set.

**Missing reference scans.** Reconstructing them later means re-finding and re-photographing every
document. Capture them at the same time.

**EXIF rotation.** Annotations drawn on the displayed orientation, model fed the stored orientation.
Silent, systematic, and affects only some photos.

**Over-fitting the generator to these 20 photos.** The calibration must *widen*. The graded set is a
different set of photos (`[REQ-49]`).

---

## Skills

- `05-skills/eval-integrity.md` — this set is the measurement instrument; protect it
- `05-skills/synthetic-data-qa.md` — for the calibration statistics

---

## Deliverables

| Artifact | Location |
|---|---|
| Real photos + reference scans | `$DATA_ROOT/real/` |
| COCO keypoint annotations | `$DATA_ROOT/real/annotations/` |
| Annotation verification figure | `outputs/figures/p01_annotations.png` |
| 5 transcripts | `$DATA_ROOT/real/transcripts/` |
| Real-photo degradation profile | `configs/real_profile.yaml` |
| `RealPhotoDataset` | `src/data/datasets.py` |
| RoboFlow link submitted | — (`[REQ-06]`, human) |
