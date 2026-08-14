# Data Contract and Intake Audit

Everything in the project sizes against data that is **not yet on this machine** (`[OPEN-01]`).
This file states what is expected, and what to check the moment it arrives.

**Run the audit before writing generator code against real data.** Assumptions made here that turn
out false will propagate into split sizes, cache budgets and the frozen-set design.

---

## 1. Assets and where they come from

| Asset | Source | Required | Status |
|---|---|---|---|
| **Clean document scans** | Teaching staff | `[REQ-01]` | `[OPEN-01]` — human has them, not on this machine |
| **Real test photos** (10–15+) | Human captures | `[REQ-02]`, `[REQ-04]` | `[OPEN-02]` |
| **Commercial reference scans** (1 per photo) | Human, via CamScanner/Adobe Scan | `[REQ-03]` | `[OPEN-02]` |
| **Corner annotations** (COCO keypoint JSON) | Human, via RoboFlow | `[REQ-05]`, `[REQ-10]` | `[OPEN-02]` |
| **Background photos** (~50) | Human captures | ADR-004 | `[OPEN-03]` |
| **DTD textures** | <https://www.robots.ox.ac.uk/~vgg/data/dtd/> | ADR-004 | Downloadable now |

---

## 2. Directory layout

Data lives outside git (`06-workflow/git-workflow.md`), under a configurable `DATA_ROOT`:

```
$DATA_ROOT/
├── scans/                    clean document scans (provided)
│   └── *.{jpg,png,tif}
├── backgrounds/
│   ├── shot/                 ~50 self-shot phone photos
│   │   └── clutter/          ≥15 hard negatives (kept separate so the ratio is measurable)
│   └── dtd/                  DTD images
├── real/
│   ├── photos/               10-15+ raw smartphone photos
│   ├── reference/            commercial app scans, filenames matching photos/
│   ├── annotations/          COCO keypoint JSON export from RoboFlow
│   └── transcripts/          hand-transcribed text for 5 documents (ADR-011 §6)
└── frozen/                   generated once, ADR-003
    ├── val/    {images/, corners.json, manifest.json}
    └── test/   {images/, corners.json, manifest.json}
```

**Filename correspondence between `real/photos/` and `real/reference/` must be exact** — the
triplet figures and the OCR comparison both depend on pairing them. Verify it in the audit, do not
assume it.

---

## 3. Expected properties — and what to do if they differ

### Clean scans

| Property | Expected | If it differs |
|---|---|---|
| Count | ~100–300 | **< 50:** flag it. The 80/10/10 split leaves ~5 test scans, which is too few for a stable test metric — escalate and consider more degradations per scan. **> 1000:** cache budget needs revisiting (ADR-003). |
| Resolution | ≥1000 px long side | **Lower:** the scan itself limits the target's sharpness; the enhancement task becomes partly impossible. Flag as a limitation for `[REQ-48]`. |
| Content | Flat, well-lit, deskewed | **Already degraded/skewed:** they are the *target*, so degradation in them caps achievable quality. Audit visually and report. |
| Colour | Mixed colour/greyscale | Fine — keep the 3-channel pipeline; greyscale scans just have equal channels. |
| Aspect | Mostly portrait A4-ish | Wide variation is fine but affects the stretch-to-square policy — check the spread. |
| Text density | Mixed | If uniformly dense, the model may not generalise to sparse pages. Note it. |

### Real photos

| Property | Expected | Check |
|---|---|---|
| Count | 10–15 (`[REQ-02]`); **20–25 recommended** | More is better; it is the only preview of reality |
| Documents | **Never seen in the scan set** | `[REQ-02]` is explicit. Verify by eye — a duplicate invalidates the generalisation claim. |
| Diversity | Lighting, viewpoint, background, camera, document type | `[REQ-04]` — audit against the checklist, not by vibes |
| Reference scan | One per photo, same document | `[REQ-03]` |

### Annotations

| Property | Requirement |
|---|---|
| Format | COCO keypoint JSON (`[REQ-10]`) |
| Count | Exactly 4 keypoints per image |
| Order | **TL, TR, BR, BL** (`[REQ-05]`, `00-project/conventions.md` §1) |
| Coverage | Every photo annotated — no gaps |
| Coordinates | Within image bounds; quad convex |

---

## 4. The intake audit — a Phase 00 gate

Produce an inventory report and write the findings to `state/discoveries.md`. Every item below is
checked, not assumed.

### Scans
- [ ] Count; total size on disk
- [ ] Resolution: min / median / max, and the distribution
- [ ] Aspect-ratio distribution (drives the stretch-to-square decision's severity)
- [ ] Channel mode: how many colour vs greyscale
- [ ] File formats present; any corrupt or unreadable files
- [ ] **Visual check of a random 20** — are they genuinely clean, flat and deskewed?
- [ ] Duplicates or near-duplicates (they would leak across the split — `[REQ-14]`)
- [ ] Text-density spread: sample a few and eyeball dense vs sparse

### Backgrounds
- [ ] Self-shot count; how many in `clutter/`
- [ ] Resolution range; confirm none are smaller than 512 on the short side
- [ ] DTD present and extracted; count matches expectation

### Real photos
- [ ] Count; every photo has a matching reference scan **by filename**
- [ ] Diversity audit against `[REQ-04]`: lighting / viewpoint / background / camera / document type
- [ ] Confirm no document appears in the scan set
- [ ] Resolution and EXIF orientation — **check for rotation metadata that `cv2.imread` ignores**;
      a photo that displays upright in a viewer but loads sideways will silently break annotation
      alignment

### Annotations
- [ ] Parse the COCO JSON; count images and keypoints
- [ ] Every image has exactly 4 keypoints
- [ ] **Render all annotations with the colour code** (`conventions.md` §8) and inspect every one.
      Ordering errors are invisible in the JSON and fatal downstream (`[REQ-05]`).
- [ ] All quads convex and correctly ordered (cross-product sign test)
- [ ] Coordinates within bounds
- [ ] Coordinate space matches the image as loaded — especially after any EXIF rotation

### Derived numbers to record
- [ ] 80/10/10 split counts, by source scan (`[REQ-14]`)
- [ ] Frozen val/test sample counts, and degradations per scan (ADR-003 targets ~500 each)
- [ ] RAM required to cache decoded scans + backgrounds at working resolution
- [ ] Per-channel mean/std of generated inputs — **training split only** (ADR-009)

---

## 5. Split construction

`[REQ-14]`: split **by source scan**, 80/10/10. `[REQ-17]`: both tasks share the same split.

- Assign each scan to exactly one of train/val/test **by a hash of its filename**, not by a shuffled
  index. A hash is stable if scans are added or the directory is re-read; a shuffle is not, and a
  quietly-changed split silently invalidates every earlier comparison.
- Write the assignment to `splits.json` and **commit it** — it is small, and it is the ground truth
  for "were these runs comparable?".
- Verify no scan appears in two splits, and check for near-duplicate scans that would leak content
  across the boundary.

---

## 6. If the data is delayed

Phase 00 and much of Phase 02 do not need the real scans. To stay unblocked:

- Build and unit-test the generator against **any** document images — a handful of PDF pages
  rendered to PNG, or public-domain scans. Structure and correctness are testable without the real
  set.
- Download DTD and start the self-shot background capture (`[OPEN-03]`).
- Capture and annotate the real photos (`[OPEN-02]`) — independent of the provided scans, and on the
  critical path for calibration.
- **Do not** freeze val/test sets or compute normalisation statistics against stand-in data. Those
  must come from the real scans, and re-freezing later invalidates comparisons.

Record clearly in `state/STATUS.md` which artifacts were built against stand-in data, so they get
re-validated on intake.
