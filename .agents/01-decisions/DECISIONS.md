# Decision Register

Every significant choice that the course spec left open has been decided deliberately, with
recorded reasoning, and written up as an ADR (Architecture Decision Record).

## Why this exists

The spec leaves genuine choices open: resolution, framework, loss weighting, background sourcing,
heatmap parameters, evaluation protocol. Without a register, an implementation agent re-derives
these ad hoc, differently each session, and the project drifts. With one, "why is it 512?" has a
one-line answer and a paragraph of reasoning behind it.

**ADRs are binding.** They are not suggestions and not history. To change one, follow the deviation
protocol in `GEMINI.md` §5: stop, write the case in `state/deviations.md`, ask the human, wait.
An approved change produces a **new ADR that supersedes the old one** — never an edit that erases
the original reasoning.

**ADRs are not requirements.** Requirements come from the graded spec and live in
`00-project/requirements.md`. An ADR is *our* decision within the freedom the spec allows. That
distinction matters when weighing a proposed change: a `[REQ]` is essentially immovable, an ADR is
movable with evidence.

## Status values

- **`ACCEPTED`** — in force.
- **`SUPERSEDED by ADR-nnn`** — replaced; kept for the reasoning trail.
- **`PROVISIONAL`** — decided, but resting on an assumption that must be validated. The validation
  point is named in the ADR and tracked in `state/assumptions.md`.

---

## The register

| ADR | Title | Status | Decides | Reversibility |
|---|---|---|---|---|
| [001](adr-001-framework-and-environments.md) | Framework, language and the three-machine setup | ACCEPTED | PyTorch; Colab-primary training; portability rules | Low — pervasive |
| [002](adr-002-resolution.md) | Working resolution: 512×512 everywhere | ACCEPTED | All three networks at 512 | Medium — retraining cost |
| [003](adr-003-data-generation-strategy.md) | On-the-fly generation + frozen eval sets + asset cache | ACCEPTED | How data reaches the GPU | Medium |
| [004](adr-004-backgrounds-and-hard-negatives.md) | Background sourcing and hard negatives | ACCEPTED | ~50 self-shot photos + DTD | High — just add data |
| [005](adr-005-enhancement-architecture.md) | Enhancement network architecture | ACCEPTED | Hand-written U-Net, 4 levels, BatchNorm | Medium |
| [006](adr-006-enhancement-loss.md) | Enhancement loss and the ablation set | ACCEPTED | 4 variants; L1+MS-SSIM expected to win | High |
| [007](adr-007-corner-approaches.md) | Both corner approaches, fairly compared | ACCEPTED | Approach A is not sandbagged | Low — REQ-30 |
| [008](adr-008-heatmap-design.md) | Heatmap representation and coordinate extraction | PROVISIONAL | σ=8 px, full-res, MSE, argmax+local soft-argmax | High |
| [009](adr-009-normalization-and-io.md) | Normalisation, value ranges, tensor conventions | ACCEPTED | Standardised input, [0,1] target, sigmoid out | Medium |
| [010](adr-010-ssim-implementation.md) | Implement SSIM/MS-SSIM ourselves | ACCEPTED | Own implementation + validation test | High |
| [011](adr-011-evaluation-and-ocr-protocol.md) | Metric definitions and the fair OCR protocol | ACCEPTED | Per-image PSNR; matched-resolution OCR; CER primary | Medium |
| [012](adr-012-bonus-scope.md) | Bonus scope: Tier 1 committed, Tier 2 optional | ACCEPTED | Chained pipeline yes; joint fine-tune conditional | High |

---

## Decisions explicitly *not* made (delegated to the implementation agent)

These were considered and deliberately left open. Choose sensibly, log what you chose in the
session log, and move on. **Do not escalate these.**

- Exact channel widths per U-Net level (a starting point is suggested in ADR-005; tune it).
- Learning rate, batch size, epoch count — within the ranges in `03-spec/training-spec.md`.
- Upsampling method: transposed conv vs `Upsample`+conv. Both are named in spec §3.1. Pick one,
  note it. (If you see checkerboard artifacts, that is your answer.)
- LR schedule: constant, cosine, or plateau-based. Keep it simple.
- Logging library or plain CSV. No preference.
- Code layout beyond `03-spec/repo-layout.md`; helper decomposition; test framework.
- How to visualise anything, beyond the fixed corner colour code in `00-project/conventions.md` §8.
- Exact parameter ranges inside each degradation, once calibrated per ADR-004 and
  `03-spec/synthetic-generator-spec.md`.

---

## Open questions

Items still needing the human are in [`open-questions.md`](open-questions.md). Check it at session
start; if an open item blocks your next action, stop and ask rather than assuming.
