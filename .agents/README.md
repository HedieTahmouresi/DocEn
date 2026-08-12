# `.agents/` — Development Environment for the Document Scanner Project

This directory is the **planning and control environment** for a university Computer Vision
project: building, from scratch, a two-network deep-learning document scanner.

It contains **no code.** It contains the requirements, the agreed decisions, the research that
justifies them, the phase plan, the working protocols, and the live project state.

---

## Who reads what

**If you are the implementation agent (Gemini):** read `GEMINI.md`, then
`state/STATUS.md`. That is the whole entry sequence. Everything else is loaded on demand.

**If you are a human picking this up:** read `00-project/project-brief.md`, then
`01-decisions/DECISIONS.md`, then `state/STATUS.md`.

**If you are a later agent inheriting the project:** `state/STATUS.md` →
`state/session-log.md` → `state/experiments.md` → `state/deviations.md`. Those four files
are designed to rebuild your context without reading any code.

---

## Setup (one time, before implementation starts)

This `.agents/` directory was authored separately from the implementation repo.

1. Copy the whole `.agents/` directory into the root of the implementation repo.
2. Copy or symlink `.agents/GEMINI.md` to `<repo-root>/GEMINI.md` so the Gemini CLI
   auto-loads it every session.
3. Place the two source documents in `.agents/00-project/source/`:
   - `Document Scanning Enhancement.md` — the official course spec (**authoritative**)
   - `Document Scanner Implementation Plan.md` — a research report (**advisory only**)
4. `git init`, commit `.agents/` as the first commit, and read
   `06-workflow/git-workflow.md` before the second.
5. Begin at `04-phases/phase-00-foundation.md`.

---

## Layout

```
.agents/
├── GEMINI.md                    ← the operating contract; loaded every session
├── README.md                    ← you are here
│
├── 00-project/                  WHAT WE ARE BUILDING AND WHAT IS FIXED
│   ├── project-brief.md         one-page orientation: goal, scope, what "done" means
│   ├── quick-reference.md       ★ every fixed number, convention, prohibition and trap, one page
│   ├── requirements.md          [REQ-nn] every hard requirement, quoted + cited
│   ├── constraints.md           [CON-nn] every prohibition, quoted + cited
│   ├── conventions.md           coordinate order, tensor layout, colour space, naming
│   ├── deliverables-checklist.md grading criteria mapped to concrete artifacts
│   └── source/                  ← place the two project documents here (see its README)
│
├── 01-decisions/                WHAT WE HAVE ALREADY DECIDED, AND WHY
│   ├── DECISIONS.md             the register — index of all ADRs, read this first
│   ├── adr-001 … adr-012        one decision each: context, options, choice, consequences
│   └── open-questions.md        unresolved items that need the human
│
├── 02-research/                 WHY THE DECISIONS ARE WHAT THEY ARE
│   ├── README.md                how to use this, and what is verified vs reported
│   ├── baseline-failure-analysis.md   the 96%-synthetic / 0%-real collapse, dissected
│   ├── sim2real-playbook.md     the domain-gap strategy; the highest-leverage part
│   ├── loss-functions.md        MSE / L1 / MS-SSIM / Sobel, with the actual paper formulation
│   ├── corner-localization.md   regression vs heatmap, sigma, extraction, literature
│   ├── evaluation-and-ocr.md    PSNR/SSIM pitfalls, corner metrics, the fair OCR protocol
│   └── source-index.md          every source cited, with what we actually took from it
│
├── 03-spec/                     HOW TO BUILD IT
│   ├── data-contract.md         what the provided dataset must contain; the intake audit
│   ├── synthetic-generator-spec.md  the degradation pipeline, exactly
│   ├── dataset-and-splits-spec.md   splits, freezing, loaders, throughput
│   ├── model-specs.md           the three network architectures
│   ├── training-spec.md         optimizers, schedules, config schema, checkpointing
│   ├── evaluation-spec.md       every number that gets reported, and how to compute it
│   └── repo-layout.md           target file structure of the implementation repo
│
├── 04-phases/                   THE PLAN, AS VERIFIABLE UNITS OF WORK
│   ├── README.md                the phase model, gate rules, dependency graph
│   └── phase-00 … phase-10      one file each: objective, tasks, gate, common failures
│
├── 05-skills/                   REUSABLE PROCEDURES
│   ├── README.md
│   ├── scope-guard.md           the anti-over-engineering / anti-drift procedure
│   ├── training-diagnostics.md  systematic debugging of a network that won't learn
│   ├── synthetic-data-qa.md     proving the generator is correct and realistic
│   ├── experiment-discipline.md how to run and compare training runs honestly
│   ├── eval-integrity.md        how not to fool yourself in the results table
│   └── portable-training.md     working across CPU box / MX330 laptop / Colab
│
├── 06-workflow/                 HOW TO OPERATE
│   ├── session-protocol.md      start-of-session and end-of-session rituals
│   ├── git-workflow.md          branches, commits, tags, what is and isn't committed
│   └── escalation-protocol.md   when to stop and ask, and how to ask well
│
└── state/                       LIVE PROJECT STATE — updated constantly
    ├── STATUS.md                ★ single source of truth: where we are, what's next
    ├── session-log.md           append-only work journal
    ├── experiments.md           registry of every training run and its verdict
    ├── discoveries.md           findings that changed our understanding
    ├── deviations.md            departures from spec/decisions + approval status
    ├── assumptions.md           unvalidated assumptions and when each must be checked
    └── templates/               copy-paste templates for the above
```

---

## The one idea to take away

The environment separates **four different kinds of statement**, and the implementation agent
is granted a different amount of freedom for each:

- **Requirements and constraints** come from the graded course spec. They are quoted verbatim
  with section citations. They are not negotiable, and nothing may be added to them.
- **Decisions** were made deliberately by the human and the planning agent, with recorded
  reasoning. They are binding until formally revised.
- **Recommendations** are research-backed suggestions. The implementation agent may exercise
  judgment, provided it logs what it chose.
- **Assumptions** are flagged as unverified, with a named point at which each must be checked.

Anything outside those four categories is ordinary engineering, and the implementation agent
should just get on with it.
