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

## Running `agy` from Claude Code

Learned the hard way, and each point cost a failed run:

```bash
agy --dangerously-skip-permissions \
    --add-dir "<your-project>" \
    --model gemini-3.8-flash-medium \
    --print "$(cat <brief path>)"
```

1. **`agy` is the first token.** No `cd <repo> &&` in front: that changes the
   command shape and the permission rule `Bash(agy *)` no longer matches. The
   repository is named with `--add-dir`.
2. **`--dangerously-skip-permissions` is mandatory.** In headless `--print`
   mode, `--mode accept-edits` silently denies every tool call and produces no
   output and no files — it looks like a hang.
3. **The brief is passed verbatim**, from a file. A correction is a new brief
   file, not text appended to the `--print` string.
4. **`agy` cannot run a long command to completion.** Its harness moves a
   long-running command into a background task and kills it when the session
   ends. It happened on both M4a runs (2026-09-22), the second time with the
   brief saying in bold to run everything in the foreground — so it is the tool,
   not the brief, and rewriting the brief again will not fix it. A brief
   therefore never asks `agy` to run a benchmark or any other slow measurement.
   The orchestrator runs it and commits the result with explicit paths. That is
   measurement, not implementation, and the PR says who ran what.

## Running `codex` beside `agy`

Authorised by the operator on 2026-09-22: `codex` is a second implementer, not
only the fallback for an exhausted Claude budget. Two briefs that do not depend
on each other run at the same time, one on each tool.

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
    -C "<your-worktree>" \
    -c model_reasoning_effort="medium" \
    -o <last-message file> - < <brief path>
```

1. **Medium effort, set per call.** The operator's `config.toml` defaults to
   `high`; the call overrides it with `-c model_reasoning_effort="medium"`. The
   config file is the operator's and is not edited.
2. **One agent, one working tree.** Two agents never share a checkout. The
   second one gets a `git worktree` (`git worktree add ../<repo>-<topic> -b
   <branch> <commit>`), and its brief names that directory, says not to switch
   branches, and names the other agent's directory as off limits.
3. **The sandbox has to be bypassed** for the same reason `agy` needs its flag:
   a worktree's git data lives in the main checkout, outside the workspace the
   sandbox allows, so commit and push fail inside it.
4. A brief that depends on an unmerged PR branches from that PR's branch and
   opens its PR against it (a stacked PR), never against `main`.

## Model policy

**The medium tier is the default for all implementation.** Not the tier that
feels safe for a given task — the medium tier, every time, with the scope made
small enough that it is sufficient.

| work | model |
|---|---|
| all implementation: contracts, backends, checks, harness, tests, README | `gemini-3.8-flash-medium`, or `gpt-oss-120b-medium` |
| architecture, licence calls, defect taxonomy, writing the briefs, reviewing PRs | Claude Opus 5, in Claude Code |
| a second, parallel implementer; and everything after the Claude budget is exhausted | `codex` at medium reasoning effort |

### Scope carries what the model does not

This is the part that makes the medium tier work, and it is the orchestrator's
job, not the implementer's. A medium model rarely fails at writing Python. It
fails at **deciding** — at choosing between two reasonable designs, at guessing
what a half-specified rule meant, at noticing that a requirement contradicts
another one three sections away.

So a brief for a medium model leaves nothing to decide:

- name the files to create, and what belongs in each;
- name the functions and their signatures, not just the behaviour;
- say what each test asserts, and what fixture it builds;
- say what is explicitly **out** of this brief, because an under-occupied model
  invents scope;
- resolve, in the brief, every question the specification leaves open — if the
  orchestrator cannot answer one, the brief is not ready to send;
- keep it to one coherent piece of work. A milestone that needs more than that
  is **split into several briefs**, in sequence.

Splitting is always the first response to a task that feels too big for the
tier. Reaching for a larger model is the last.

### Escalation

Escalate one tier only when a PR has failed review **twice on the same point**,
and record it in the PR. Two failures on the same point means the brief was
ambiguous, so rewriting the brief is tried before the model is changed —
usually it is the cheaper and the more durable fix.

Going straight to the expensive model "to be safe" is the exact failure this
section exists to prevent. `gemini-3.1-pro-high` and the thinking Claude models
are available and are the default for nothing.

### Recorded deviations

**M2 — presence.** Implemented on `claude-sonnet-4-6` before this rule was
tightened, on the reasoning that the presence logic needed judgement. Under the
rule above that was the wrong call: the judgement should have gone into the
brief instead. M3 onwards runs on the medium tier.

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

- **Free inside `<your-project>`**: create, edit and delete files, run
  tests, install permissive dependencies, commit, push, open PRs, and **merge
  its own PR into `main` once its tests pass**.
- **Every change still goes through a PR.** Never commit to `main` directly. The
  PR is the whole audit trail — it must state what was built, what it was tested
  against, and what it could not finish. A merged PR with an empty description
  defeats the only control we kept.
- **Source runs are read-only, always.** Runs under `build/runs/**` are opened
  for reading and nothing else. No image, audio file or run artefact is ever
  copied into this repository (`AGENTS.md`).
- **Never relax a licence rule.** The rules in `AGENTS.md` §licensing are not
  negotiable by an implementer. Blocked by one → say so in the PR and stop.
- **Never invent a milestone.** Build the one in the brief. A good idea found on
  the way is written into the PR as a suggestion, not implemented.

## This repository is public

`Craft-dreams/frame-jury` is public, and an implementer here can merge its own
work. So a mistake is not a local mistake — it is published, and a pushed secret
stays in the history after the file is deleted.

- **No key, token or credential in the repository, ever.** Not in code, not in a
  test, not in a README example, not in a commit message, not in a `.env` that
  someone forgets is tracked. The `.env` pattern is already in `.gitignore`;
  keep it there.
- The optional VLM backend (`budget=full`, SPEC §6) is the one component that
  needs a credential. It reads it from the environment at call time, has no
  default, and **skips its tests when the variable is absent** rather than
  carrying a fallback value. No test may require a key to pass, which the
  existing rule that tests need no network already implies.
- Example configuration is written with an obvious placeholder
  (`FRAME_JURY_VLM_API_KEY=<your key>`), never a real-looking string.
- Nothing about the operator's runs is published: no image, no run path, no
  case file. Absolute machine paths live in gitignored case files, not in
  committed code or documentation.
- If a secret is ever pushed, say so immediately in the PR and treat the key as
  burned — rotate it. Deleting the file does not remove it from the history.

## Why the labels are not blocking this

The operator has no desk time to label, so the 300 labels M1 needs do not exist.
Rather than idle, every milestone is built to the seam where labels are actually
required — the *fitting* of thresholds — and stops there behind a default
threshold file. M7 then makes labelling possible from a phone on a commute,
which is the real unblocker. See `SPEC.md` §9.

This means numbers are not trustworthy until the labels land. **No quality claim
may be made from an unlabelled corpus**, and no milestone may be called done on
the strength of one.
