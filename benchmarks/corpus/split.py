"""Create deterministic dev/test corpus splits with run-level isolation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import iter_cases

SPLIT_SCHEMA_VERSION = "1.0"


def split_cases(
    cases: list[dict[str, Any]], *, seed: int, test_fraction: float = 0.2
) -> dict[str, Any]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be greater than 0 and less than 1")
    by_run: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_run[case["provenance"]["run_id"]].append(case["case_id"])
    run_ids = sorted(by_run)
    shuffled = run_ids.copy()
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        test_count = 0
    else:
        test_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * test_fraction)))
    test_runs = set(shuffled[:test_count])
    dev_runs = set(run_ids) - test_runs

    def partition(selected: set[str]) -> dict[str, list[str]]:
        ordered_runs = sorted(selected)
        return {
            "run_ids": ordered_runs,
            "case_ids": sorted(case_id for run_id in ordered_runs for case_id in by_run[run_id]),
        }

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "test_fraction": test_fraction,
        "dev": partition(dev_runs),
        "test": partition(test_runs),
    }


def write_split(path: str | Path, split: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(split, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="validated cases JSONL")
    parser.add_argument("--out", required=True, type=Path, help="destination splits.json")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cases = list(iter_cases(args.cases))
        split = split_cases(cases, seed=args.seed, test_fraction=args.test_fraction)
        write_split(args.out, split)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {len(split['dev']['case_ids'])} dev and "
        f"{len(split['test']['case_ids'])} test cases to {args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
