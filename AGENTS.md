# frame-jury — working rules for anyone building here, human or agent

Read `SPEC.md` first. It is the contract: purpose, JSON interface, defect
taxonomy, architecture, corpus, benchmark protocol and the bar that must be
cleared. This file is how the work is done, not what is built.

## The one idea this project must not lose

frame-jury judges a frame **against the declaration that asked for it**: which
entities must be visible, of what kind, what the character looks like, how the
shot is framed, and what the prompt asked for and forbade. Detectors provide
evidence; the declaration decides what the evidence means. Any design that
drifts toward "a better detector, applied to an image" has lost the point.

## Licensing, which is a hard boundary and not a preference

The project is AGPL-3.0-or-later with a commercial licence sold beside it, so
the package must stay relicensable:

1. `frame_jury/` may depend only on MIT, BSD or Apache-2.0 code **and weights**.
   No AGPL. No research-only or non-commercial weights. No "free for
   non-commercial use" model, whatever its quality.
2. Competitor adapters live only in `benchmarks/competitors/`, are an optional
   extra (`pip install .[competitors]`), and are never imported by `frame_jury`.
   A test enforces this by importing every module of the package with the
   competitor packages made unavailable.
3. Model weights are never committed or vendored. They are downloaded from their
   publisher, pinned by sha256, and recorded in `THIRD_PARTY_NOTICES.md` with
   licence and URL.
4. HADM (`wangkaihong/HADM`) publishes no licence: no rights are granted. Do not
   copy its code, do not download its weights, not even for benchmarks. Its
   published ideas may be reimplemented from the paper, with the citation in the
   code.
5. When reproducing any donor's approach, cite the paper or repository and its
   licence in a comment, and write the implementation from the description.
   Copying permissively licensed code is allowed only with its notice preserved
   and an entry in `THIRD_PARTY_NOTICES.md`.

If a rule here blocks something worth doing, say so in the pull request and stop
there. Do not route around it.

## The corpus is borrowed, and read-only

Cases are built from external generator runs (`build/runs/**`). Those
directories are **input, opened read-only**. Never write into them, never move
them, and never commit an image, an audio file or a run artefact into this
repository. The corpus builder emits case files that *point at* paths; labels
are stored here, images are not.

## How work lands

One milestone, one pull request, reviewed before the next begins. `SPEC.md` §9
lists them. A pull request is ready when:

- the feature has tests that fail without it;
- tests need no network and no GPU; a CPU path always exists;
- anything measurable comes with the benchmark row that shows it, and the
  leaderboard file is updated in the same commit;
- the README tells a person on a clean machine how to run what you added;
- determinism holds: same input, same verdict, with pinned weights and seeds.

Write commit messages that say what changed and why it was needed. A message
that only names the files is not useful to whoever reads it in six months.

## Judging honestly

The benchmark exists to find out whether this is any good, not to prove that it
is. So:

- keep a `null` baseline (accept everything) and report it; anything that cannot
  beat it is not working;
- fit thresholds on the dev split only, and never look at the test split while
  tuning;
- split by run, so frames from the same film cannot sit on both sides;
- report precision *and* recall per defect, with the count of cases behind each
  number. A percentage over nine examples is noise, and should be printed as
  such;
- when a competitor wins, say it plainly in the pull request. That result is the
  most valuable thing this repository can produce.

## Reporting

At each milestone report: what was built, the leaderboard diff, what surprised
you, and what you would change in `SPEC.md`. The specification is expected to be
wrong somewhere; finding where is part of the work.
