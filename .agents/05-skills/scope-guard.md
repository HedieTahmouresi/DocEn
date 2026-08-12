# Skill: Scope Guard

**Load before:** adding any component, dependency, loss term, architectural feature or training
technique that is not already in `03-spec/`. **Mandatory before any architecture change.**

---

## Why this exists

The literature contains a better version of every component in this project. Diffusion-based
document enhancement, attention-gated skip connections, transformer U-Nets, adaptive losses,
corner-refinement cascades, GAN sharpening. Each is real, each works, and each is **out of scope**.

The failure this guards against is not laziness — it is the opposite. It is a capable agent
improving the system past the point where it can still be finished, defended, and graded. The
research report itself reaches this conclusion: implementing diffusion models or attention-gated
skips "falls into the realm of extreme overengineering for this specific project parameter."

There is a second, subtler failure: **drift**. Changing an agreed design mid-implementation because
something else seems better in the moment. That is not over-engineering, it is a broken contract —
and it destroys the ability of anyone (including you, next session) to reason about why the system
is the way it is.

---

## The procedure

Run this before adding anything. It takes a minute.

### Step 1 — Is it required?

`grep` `00-project/requirements.md` for it.

- **Found, with a citation** → it is required. Build it. Stop here.
- **Not found** → it is not a requirement, no matter how strongly it feels like one. Continue.

> **The most common error in this project:** treating a statement from the research report
> (`Document Scanner Implementation Plan.md`) as a requirement. That document is advisory and
> contains at least one direct contradiction of the spec. Only `requirements.md` is authoritative.

### Step 2 — Is it forbidden?

Check `00-project/constraints.md`. The traps that catch people:

- Perceptual / VGG / LPIPS loss → **pretrained weights** → `[CON-02]`
- Any `torchvision.models` or `timm` backbone, even untrained → `[CON-01]`
- `albumentations` / `imgaug` / `kornia.augmentation` in the degradation pipeline → `[CON-03]`
- `weight_decay > 0` or any dropout before Phase 07 → `[CON-04]`
- `AdamW` (defaults to `weight_decay=0.01`) → `[CON-04]`
- Any flip augmentation → `[CON-05]`

If forbidden: **stop.** Do not look for a workaround. Note it in the report's "what we could not
try" — that is worth marks.

### Step 3 — Does it contradict a decision?

Check `01-decisions/DECISIONS.md`. If an ADR covers it, the ADR is binding.

- Want to change it? → **deviation protocol** (`GEMINI.md` §5). Stop, write the case, ask, wait.
- Not covered? → continue.

### Step 4 — The four questions

Answer all four honestly. **Any "no" means don't build it.**

1. **Does this address a problem I have actually measured?**
   Not anticipated, not read about. Measured, with a number, in `state/experiments.md`.
2. **Is it the simplest thing that addresses it?**
   If a config change or more data would do the same job, do that instead.
3. **Can I explain it at the presentation without reading the code?**
   `[REQ-43]`: "be prepared to explain and modify any part of the code if asked." Complexity you
   cannot defend is negative value.
4. **Does it fit in the remaining time without threatening a mandatory deliverable?**
   Bonus work must never displace required work (ADR-012).

### Step 5 — If all four pass

Build it — as a **separate, logged experiment** (`experiment-discipline.md`), with the baseline
preserved so the comparison is real. Record what you chose and why in the session log.

---

## Project-specific temptation list

Things you will genuinely be tempted by, and what to do instead.

| Tempted to add | Why it's tempting | Verdict | Do instead |
|---|---|---|---|
| Attention-gated skip connections | The "semantic gap" in U-Net skips is a real, published phenomenon | **No** — ADR-005, `[CON-10]` | Plain concat skips; mention it in `[REQ-48]` |
| Transformer / ViT blocks | State of the art | **No** — `[CON-10]` | 4-level conv U-Net |
| Diffusion (DocDiff-style) | The current SOTA for document enhancement | **No** — far out of scope | Note in limitations |
| GAN / adversarial loss | Would give the sharpest text | **No** — instability, second failure mode | L1 + MS-SSIM (ADR-006) |
| Perceptual / VGG / LPIPS loss | The standard answer for sharpness | **Forbidden** — `[CON-02]` | Say so in the report |
| Adaptive Wing Loss | Genuinely better for heatmaps; real published gains | **Escalate** — ADR-008 | Foreground-weighted MSE first (pre-approved) |
| Corner-refinement cascade | Published, improves sub-pixel accuracy | **No** — 4 extra passes for accuracy we don't need | Local soft-argmax (ADR-008) |
| Deeper / wider U-Net | "Maybe it's underfitting" | **Measure first** | Check training metrics; if training itself is poor, it's underfitting — otherwise it isn't |
| More degradation types | More realism must be better | **No** — `[REQ-34]` fixes the six | Widen the *ranges* of the existing six |
| Test-time augmentation | Free accuracy | **No** — not in scope, complicates the pipeline | — |
| Model ensembling | Free accuracy | **No** — doubles inference, unexplainable gain | — |
| EMA of weights / SWA | Standard modern practice | **No** — arguably `[CON-04]` regularisation, and unnecessary | Cosine schedule |
| Elaborate LR schedules | Squeeze out a bit more | **No** | Cosine or constant |
| Curled-page dewarping | Real photos have curl | **No** — explicitly out of scope | `[REQ-48]` limitation |
| Multi-document detection | Robustness | **No** — one document per image | — |
| A second OCR engine | Cross-check robustness | **Optional** — only if Phase 05 finishes early | Tesseract is named in the spec |
| Predicting the residual | Often speeds up restoration convergence | **Allowed** — ADR-005 `[REC]` | But only *after* the plain version passes its gate, as a logged experiment |
| Distractor quadrilaterals | Attacks the known failure mode | **Allowed as an ablation** — ADR-004 | Baseline first, then measure |

---

## When performance disappoints, in priority order

The most common over-engineering trigger is a disappointing number, and the instinct is to reach for
architecture. For this project that is almost always wrong.

**Corner detection weak on real photos:**
1. Background diversity and clutter ← **start here**
2. Homography range width (the baseline's exact failure)
3. Photometric range width
4. Heatmap σ / loss weighting
5. Architecture ← almost never the answer

**Enhancement weak on real photos:**
1. Degradation realism, especially illumination and shadows
2. The stranger test — what gives your synthetics away?
3. Loss weighting
4. Architecture ← last

**Everything weak, synthetic included:** this is not a domain gap. It is a bug or under-training.
Go to `training-diagnostics.md`.

---

## The anti-drift rule

> If you find yourself implementing something different from what `03-spec/` describes, **stop**.
> Either you have found a real problem with the spec — which is a deviation, and needs the human —
> or you are drifting.

Drift is seductive because each individual step is reasonable. The cumulative effect is a system
nobody decided to build, whose rationale exists only in a context window that no longer exists.

If a spec document is genuinely wrong, that is valuable information. Raise it. Do not route around
it silently.
