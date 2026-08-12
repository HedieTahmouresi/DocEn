# Template: Session Log Entry

Copy into the **top** of `state/session-log.md` (newest first).

**The test:** could another agent, knowing nothing about your session, continue from this?

---

```markdown
## YYYY-MM-DD — Phase NN: <short subject>

**Did:** what you actually built or attempted. Concrete, not "worked on the generator".

**Result:** what happened, with numbers. Gate outcomes go here — including failures, with the
actual output.

**Learned:** anything that changes understanding. If it changes the *plan*, also add it to
`discoveries.md`.

**Decided:** choices made where the environment left them open (`[REC]` items, engineering calls).
Say what you chose and why — this is how the next agent avoids re-litigating it.

**Blocked / parked:** anything waiting on the human or on another phase, and what you did instead.

**Next:** the specific next action. Must match `STATUS.md`.

**Commits:** short hashes.
```

---

## Worked example

```markdown
## 2026-08-14 — Phase 02: degradation steps 1-3

**Did:** implemented the perspective warp with shape-then-place corner sampling, resolution loss,
and photometric adjustment. Added the degeneracy rejection loop (convexity + min interior
angle 20 deg).

**Result:** round-trip alignment PSNR = 34.2 dB with photometrics off — above the 30 dB gate.
Corner overlays verified on 20 samples including extreme geometry (near-zero margin, max
perspective). Parameter histograms show no spikes or clipping.

**Learned:** the rejection loop fires on ~3% of samples at perspective_strength up to 0.35 — low
enough not to bias the distribution. Above 0.45 it climbs past 15%, which *would* reshape it, so
0.35 stays the ceiling. Noted in discoveries.md.

**Decided:** INTER_AREA for downscale, randomised interpolation for upscale. Not specified
anywhere — [REC]-level engineering call.

**Blocked / parked:** nothing. Backgrounds still DTD-only ([OPEN-03]); calibration ranges remain
provisional until the human's photos arrive.

**Next:** illumination gradient and shadow compositing, per synthetic-generator-spec.md §8. This is
the most important degradation for the enhancement task — randomise gradient direction, steepness,
shadow count/shape/blur/opacity, and shadow *presence*.

**Commits:** a1b2c3d, e4f5g6h
```

---

## What makes an entry weak

| Weak | Why |
|---|---|
| "Worked on the generator" | Not resumable — what part, what state? |
| "Fixed some bugs" | Which bugs? Would the next agent recognise them recurring? |
| "Training is running" | Which experiment? Where? What is the expected result? |
| "Next: continue" | Forces the next agent to re-derive the plan |
| No numbers | Gate outcomes and results are the point |
| Omitting a failure | The most valuable thing to record; it stops a repeat |
