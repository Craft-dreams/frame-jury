# frame-jury

Judges a generated frame against the contract that asked for it.

Every public tool for this judges an image blind: one compares two faces,
another counts what it recognises, another finds broken limbs, a vision model
gives an opinion. None of them knows what the image was *supposed* to show — so
each one has to guess whether two people in a frame are a mistake.

frame-jury starts from the declaration instead. A shot says which entities must
be visible, what kind each one is, what the character looks like (an approved
reference image), how it is framed, and what the prompt asked for and forbade.
Against that, "two people where one was declared" is not a guess: it is a
contract violation, with the numbers to prove it.

It uses detectors as evidence, not as opinions, and it says `unsure` when the
evidence does not settle the question — a judge that guesses is worse than one
that abstains.

## Status

Specification first. `SPEC.md` is the contract, the architecture, the defect
taxonomy, the benchmark protocol and the bar that must be cleared before this is
plugged into a production pipeline. Code follows it, milestone by milestone.

## What it does

```text
how many people are in this frame     -> permissive detector, on CPU, ~100 ms
is this the same character as before  -> reference image vs frame
are the hands and limbs broken        -> research track
does the frame show what was asked    -> a vision model, last and dearest
```

Cheap and deterministic checks run first and gate the expensive ones. Thresholds
are calibrated data per camera framing, not numbers typed into the code.

## Licence

**AGPL-3.0-or-later**, with a commercial licence available.

Using frame-jury inside software you offer to others — including over a network —
places that software under the AGPL. If that does not suit your product, a
commercial licence removes the obligation: see `COMMERCIAL-LICENSE.md`.

The commercial licence covers the `frame_jury` package, whose dependencies and
model weights are deliberately permissive (MIT, BSD, Apache-2.0) so that it can
be relicensed. It does **not** cover `benchmarks/competitors/`, which exists to
measure other projects against ours and imports AGPL-licensed code; those
adapters are AGPL-only and are never part of a release.

Contributions require signing the CLA in `CLA.md`, which is what keeps the dual
licence possible.

Every model the package downloads is recorded in `THIRD_PARTY_NOTICES.md` with
its licence and the checksum of its weights.
