# Source Documents

**Place the two project documents in this directory before implementation starts.**

They are not committed here because this `.agents/` environment was authored separately from the
implementation repository. Copy them in during Phase 00 setup.

| File | Authority |
|---|---|
| `Document Scanning Enhancement.md` | **AUTHORITATIVE** — the official course specification. This is what is graded. Every `[REQ-nn]` and `[CON-nn]` derives from it, with section citations. |
| `Document Scanner Implementation Plan.md` | **ADVISORY ONLY** — a research report. Useful analysis, but not binding, and it contains errors. |

---

## Why the distinction is enforced

`00-project/requirements.md` and `00-project/constraints.md` are the **distilled, cited** form of
the specification. In normal work you read those, not the raw document — they are indexed, tagged,
and traceable.

Come back to the original when:
- You need the full context around a citation
- A requirement's wording is ambiguous and the surrounding text would settle it
- You suspect the distillation missed something (**if so, that is a finding — record it in
  `state/discoveries.md` and update `requirements.md`**)

**Never treat the research report as a source of requirements.** It is analysis, and it is
sometimes wrong.

---

## Known errors in the research report

Recorded in `02-research/README.md`. The one that matters most:

> The report's Phase 2 instructs "strictly abandoning the direct regression methodology" for corner
> detection. The course specification **mandates implementing both** Approach A and Approach B and
> comparing them empirically — `[REQ-30]`, spec §5: "you will implement both and let the experiments
> decide which one wins."
>
> Approach B may *win*. It may not be *skipped*.

Also: unverified empirical claims about a baseline notebook we do not possess, an irrelevant passage
about UK financial-sector document standards, no notice of the OCR resolution ceiling, and a
library recommendation that ADR-010 declined.

Where the report is **right** and genuinely valuable — the baseline failure analysis, hard-negative
backgrounds, 512 over 256, the L1+MS-SSIM loss, rejecting over-engineered alternatives — is also
listed in `02-research/README.md`.

---

## A note on the specification file

`Document Scanning Enhancement.md` is ~386 KB, but only the **first ~370 lines** are text. The
remainder is base64-encoded figures on four very long lines. Read with a line limit rather than
loading the whole file, or a read will fail on size.

The embedded figures are: the title image, the U-Net architecture diagram (§3.1), the PSNR/SSIM
formulas (§3.3), and the corner-detection approaches diagram (§5).
