# Assumption Ledger

Things believed but **not verified**, each with a named point at which it must be checked.

**Why this file exists:** decisions built on unvalidated assumptions are fragile in a specific,
dangerous way — they look like decisions. Tagging them `[ASM]` and naming the validation point makes
the fragility visible, so that when an assumption turns out false, the affected decision is
immediately identifiable.

**The rule:** when validating an assumption, update this file **whether it held or not**. A
falsified assumption is more valuable than a confirmed one, and it triggers a review of everything
built on it (`GEMINI.md` §6 — a false `[ASM]` is an escalation trigger).

---

## Status values

`OPEN` — not yet checked · `VALIDATED` — confirmed, with evidence ·
`FALSIFIED` — wrong; review dependents · `SUPERSEDED` — no longer relevant

---

## Register

| ID | Assumption | Validate at | Status | Affects |
|---|---|---|---|---|
| `[ASM-01]` | Colab T4 is ~15–20× faster than the MX330 on this workload | Phase 00 §E benchmark | OPEN | ADR-001 machine roles |
| `[ASM-02]` | The provided scan set is ~100–300 scans | Phase 00 intake audit | **FALSIFIED (50 scans)** | Split counts: 41/5/4; updated held-out samples |
| `[ASM-03]` | Provided scans are genuinely clean, flat and deskewed | Phase 00 visual audit | **VALIDATED** | Achievable enhancement quality ceiling |
| `[ASM-04]` | Zhao et al.'s α=0.84 transfers from natural images to documents | Phase 04 α sweep | OPEN | ADR-006 loss weighting |
| `[ASM-05]` | σ=8 px at 512 is a reasonable heatmap width | Phase 06 σ sweep | OPEN | ADR-008 (marked PROVISIONAL) |
| `[ASM-06]` | Plain MSE trains the heatmap network adequately | Phase 06 first run | OPEN | ADR-008 loss; weighted MSE is the pre-approved fallback |
| `[ASM-07]` | ~2 vCPUs on Colab can feed a T4 after ADR-003's optimisations | Phase 03 GPU-utilisation gate | OPEN | ADR-003; the whole training schedule |
| `[ASM-08]` | Free-tier Colab quota suffices for the full 512 matrix | Continuous | OPEN | ADR-002, `[OPEN-08]` |
| `[ASM-09]` | BatchNorm is acceptable under `[CON-04]`'s "explicit regularization" | Ask a TA if possible | OPEN | ADR-005, `[OPEN-09]` |
| `[ASM-10]` | The baseline's reported figures (96%/0.00%, 1.85/107.44 px) are accurate | **Cannot verify — notebook unavailable** | OPEN | Only the *pattern* is relied on, never the magnitudes |
| `[ASM-11]` | 20–25 real photos are enough to estimate the real distribution for calibration | Phase 01 coverage plot | OPEN | ADR-004 §3 calibration; mitigated by the deliberate widening |
| `[ASM-12]` | All four page corners visible in every real photo (no truncation) | Phase 01 annotation | OPEN | ADR-004 §4 — truncation is out of scope by design |
| `[ASM-13]` | Documents in the real photos never appear in the provided scan set | Phase 00 / Phase 01 audit | **VALIDATED** | `[REQ-02]`; the generalisation claim depends on it |

---

## Notes on the higher-consequence ones

**`[ASM-02]` — scan count.** The most widely-depended-upon assumption here. Split sizes, frozen-set
sizing (~500 val / ~500 test, i.e. ~25 degradations per held-out scan) and the RAM cache budget all
assume ~200 scans.
- **Under 50** → ~5 test scans, too few for a stable test metric. Escalate; consider more
  degradations per scan and state the limitation.
- **Over 1000** → revisit the cache budget (ADR-003), especially on the 7 GB workstation with
  4 workers.

**`[ASM-07]` — CPU throughput.** The main technical risk of the compute setup. If it fails, the
options ladder is in ADR-003, and the last rung touches `[REQ-11]` — which makes it a deviation, not
an optimisation. **Measure at the Phase 03 gate; do not assume and launch.**

**`[ASM-10]` — the baseline's numbers.** The underlying notebook is not in our possession. Everything
in this environment relies on the *pattern* (a high synthetic score coexisting with total real-world
failure, traced to fixed generator parameters), never on the magnitudes. Do not benchmark against
1.85 px or 107.44 px — the resolution and threshold behind them are undocumented.

**`[ASM-09]` — BatchNorm.** Low risk; the reading in ADR-005 is standard. But if it is ever going to
be challenged, **finding out early is much cheaper** — it invalidates trained checkpoints. Ask a TA
if there is any opportunity.

**`[ASM-11]` — calibration sample size.** 20–25 photos is a thin estimate of a distribution. This is
exactly why ADR-004 §3 widens the ranges by 1.5–2× rather than matching them: the widening absorbs
the estimation error, and the graded set is a different set of photos (`[REQ-49]`).

---

## Adding an assumption

Whenever you write "assuming…", "presumably…", "this should be…", or build on a number you did not
measure — add it here with the next `[ASM-nn]`, a validation point, and what it affects.

An assumption with no validation point is not an assumption, it is a belief. Give it one.
