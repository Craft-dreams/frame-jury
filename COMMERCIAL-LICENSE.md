# Commercial licence

frame-jury is published under the **GNU Affero General Public License v3.0 or
later** (see `LICENSE`). The AGPL asks something specific in return: if you run
a modified or unmodified version inside software you offer to other people —
including as a hosted or network service — the source of that software must be
offered to its users under the AGPL as well.

That suits research, internal tools and open products. It does not suit a closed
commercial product, and it is not meant to be a trap: a **commercial licence** is
available, which grants the same software without the AGPL obligations.

## Who needs one

You need a commercial licence if you want to:

- embed frame-jury in a proprietary product or service;
- offer it, or something built on it, over a network without releasing your
  source under the AGPL;
- distribute it inside software you do not wish to license under the AGPL.

You do **not** need one to evaluate it, to use it internally, to do research, or
to build open-source software licensed under the AGPL.

## What it covers

The `frame_jury` package and its released artefacts. Its dependencies and model
weights are kept to permissive licences (MIT, BSD-3-Clause, Apache-2.0) on
purpose, so that the package can be licensed commercially without anyone else's
terms coming along.

It does not cover `benchmarks/competitors/`, which imports third-party projects
under their own licences — including AGPL-3.0 — in order to measure them. Those
adapters are for evaluation only and are not part of any release.

Model weights are never relicensed by us. The package downloads them from their
publishers under their own terms, listed in `THIRD_PARTY_NOTICES.md`.

## How to obtain one

Open an issue titled "commercial licence" in this repository, or contact the
maintainers of Craft-dreams. Say what you are building and how frame-jury would
be used in it; terms are agreed case by case.

## Copyright

Copyright (c) 2026 Craft-dreams. frame-jury is dual-licensed: AGPL-3.0-or-later
for everyone, and commercially by agreement. The copyright in contributions is
handled by `CLA.md`, which is what makes the second licence possible.
