# Delegation — who builds what, on which model, and with what freedom

`SPEC.md` says what is built. `AGENTS.md` says how work lands. This file says
**who does it and what it costs**, because the budget is a design constraint
here: the orchestrating model is the expensive one, and spending it on
mechanical implementation is how a project stalls halfway.

Decided 2026-09-21 with the operator.

## The shape

```text
Claude Opus 5 (Claude Code)          orchestrates: specification, decomposition,
                                     review of every PR, licence judgement
        │  writes the milestone brief
        ▼
agy --dangerously-skip-permissions   implements: one milestone, one branch,
  (medium model by default)          tests, README, PR
        │  PR
        ▼
Claude Opus 5                        reviews, merges, writes the next brief
```

The orchestrator never writes the implementation, and the implementer never
decides the architecture. When the implementer thinks the specification is
wrong, it says so in the PR and stops — it does not route around it
(`AGENTS.md`).

## Model policy

The rule is **cheapest model that can do the task, escalate on evidence, never
escalate on a hunch.** This mirrors the factory's own rule that expensive work
follows cheaper gates.

| work | model | why |
|---|---|---|
| mechanical implementation from a written brief: backends, adapters, plumbing, tests, README | `gemini-3.8-flash-medium` | the brief already contains the decisions; this is typing, and it is the bulk of the milestones |
| implementation needing judgement: the presence check's logic against the declaration, the identity abstain rule, the harness protocol | `claude-sonnet-4-6` | wrong judgement here is expensive to find later, and it is a small share of the lines |
| architecture, licence calls, defect taxonomy, reviewing PRs, deciding what "done" means | Claude Opus 5, in Claude Code | these are the decisions the whole repository is built on |
| anything after the Claude budget is exhausted | `codex` | the fallback, see below |

Escalate a task one tier when, and only when: the cheap model produced a PR that
failed review twice on the same point, or the task turned out to need a decision
the brief did not contain. Record the escalation in the PR. Going straight to
the expensive model "to be safe" is the failure this table exists to prevent.

`gpt-oss-120b-medium` is the alternate for the cheap tier if Gemini Flash proves
weak on Python; `gemini-3.1-pro-high` is available but is not the default for
anything, because the medium tier has not yet been shown to fail.

## Token exhaustion, which is expected and planned for

When the Claude budget runs out mid-milestone, work continues on `codex` with
the same brief. This survives a handover because **the brief, not the
conversation, is the unit of delegation**: every milestone brief is written into
the PR description before implementation starts, so a new agent on a different
vendor picks it up from the repository alone.

Handover checklist, in order:

1. The current branch is pushed, even unfinished, with its tests in whatever
   state they are.
2. The PR description carries the brief and a "stopped here" line.
3. `codex` is started in the repository root with the PR number and told to
   continue from the brief.
4. The orchestrator still reviews. Review is never delegated to the fallback,
   because review is the one thing that keeps a cheap implementer honest.

## What the implementer may do without asking

Agreed with the operator on 2026-09-21. This is not production code and the
priority is finishing the component, so autonomy is wide and traceability is
what is preserved instead:

- **Free inside `Projetos\frame-jury`**: create, edit and delete files, run
  tests, install permissive dependencies, commit, push, open PRs, and **merge
  its own PR into `main` once its tests pass**.
- **Every change still goes through a PR.** Never commit to `main` directly. The
  PR is the whole audit trail — it must state what was built, what it was tested
  against, and what it could not finish. A merged PR with an empty description
  defeats the only control we kept.
- **`content-factory` is read-only, always.** Runs under `build/runs/**` are
  opened for reading and nothing else. No image, audio file or run artefact is
  ever copied into this repository (`AGENTS.md`).
- **Never relax a licence rule.** The rules in `AGENTS.md` §licensing are not
  negotiable by an implementer. Blocked by one → say so in the PR and stop.
- **Never invent a milestone.** Build the one in the brief. A good idea found on
  the way is written into the PR as a suggestion, not implemented.

## Why the labels are not blocking this

The operator has no desk time to label, so the 300 labels M1 needs do not exist.
Rather than idle, every milestone is built to the seam where labels are actually
required — the *fitting* of thresholds — and stops there behind a default
threshold file. M7 then makes labelling possible from a phone on a commute,
which is the real unblocker. See `SPEC.md` §9.

This means numbers are not trustworthy until the labels land. **No quality claim
may be made from an unlabelled corpus**, and no milestone may be called done on
the strength of one.
