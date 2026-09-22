"""benchmarks.run — benchmark harness, timing protocol, and leaderboard writer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.competitors import Competitor, CompetitorResult, FrameJuryCompetitor, NullCompetitor
from benchmarks.corpus.schema import iter_cases
from benchmarks.label.schema import DEFECTS
from benchmarks.scoring import score

PINNED_PACKAGES = (
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "pillow",
    "torch",
    "torchvision",
)


def get_pinned_versions() -> dict[str, str]:
    """Return dictionary of Python version and installed package versions."""
    versions: dict[str, str] = {
        "Python": sys.version.split()[0],
    }
    for pkg in sorted(PINNED_PACKAGES):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def load_labels(path: str | Path | None) -> dict[str, set[str]]:
    """Load ground-truth labels from JSONL, tolerating non-existent file."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    labels: dict[str, set[str]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            defects = set(data.get("defects", []))
            if data.get("taxonomy_version") == "2.0" and "broken_body" in defects:
                defects.add("broken_body_v2_0")
            labels[data["case_id"]] = defects
    return labels


def render_leaderboard(
    *,
    split_name: str,
    case_count: int,
    labelled_count: int,
    excluded_legacy_count: int,
    timestamp: str,
    pinned_versions: dict[str, str],
    rows: list[dict[str, Any]],
) -> str:
    """Render deterministic markdown leaderboard text."""
    lines: list[str] = [
        "# Benchmark Leaderboard",
        "",
        f"- **Split**: {split_name}",
        f"- **Cases**: {case_count}",
        f"- **Labelled**: {labelled_count}",
        f"- **Excluded legacy (`broken_anatomy`)**: {excluded_legacy_count}",
        f"- **Timestamp**: {timestamp}",
        "- **Pinned versions**:",
    ]
    for pkg in sorted(pinned_versions.keys()):
        lines.append(f"  - {pkg}: {pinned_versions[pkg]}")

    lines.extend(
        [
            "",
            "| system | defect | precision | recall | F1 | abstain % | ms/frame | peak RAM (py) | licence |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    # Sort rows by system then defect
    sorted_rows = sorted(rows, key=lambda r: (r["system"], r["defect"]))

    for r in sorted_rows:
        prec_str = f"{r['precision']:.3f}" if r["precision"] is not None else "—"
        rec_str = f"{r['recall']:.3f}" if r["recall"] is not None else "—"
        f1_str = f"{r['f1']:.3f}" if r["f1"] is not None else "—"
        abs_str = f"{r['abstain_pct']:.3f}" if r["abstain_pct"] is not None else "—"
        ms_str = f"{r['ms_per_frame']:.3f}"
        ram_str = f"{r['peak_ram_mb']:.3f}"
        lines.append(
            f"| {r['system']} | {r['defect']} | {prec_str} | {rec_str} | {f1_str} | {abs_str} | {ms_str} | {ram_str} | {r['licence']} |"
        )

    lines.extend(
        [
            "",
            "*peak RAM (py) measures Python allocations only (stdlib tracemalloc in MB).*",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    cases_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    split: str = "test",
    out_path: str | Path = "benchmarks/LEADERBOARD.md",
    runs: int = 3,
    competitors: list[Competitor] | None = None,
    timestamp: str | None = None,
) -> str:
    """Run the benchmark harness, evaluate competitors, and write the leaderboard."""
    cases_file = Path(cases_path)
    if not cases_file.exists():
        msg = f"Cannot run benchmark: cases file does not exist: {cases_file}"
        print(msg, file=sys.stderr)
        raise FileNotFoundError(msg)

    all_cases = list(iter_cases(cases_file))

    # Split filtering
    if splits_path is not None and Path(splits_path).exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        if split not in splits_data:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(splits_data.keys())}")
        selected_ids = set(splits_data[split]["case_ids"])
        cases = [c for c in all_cases if c["case_id"] in selected_ids]
    else:
        cases = all_cases

    # Verify images exist
    missing_images: list[str] = []
    for c in cases:
        img_path = Path(c["image_path"])
        if not img_path.exists():
            missing_images.append(str(img_path))

    if missing_images:
        msg = (
            f"Cannot run benchmark: {len(missing_images)} of {len(cases)} image(s) "
            f"do not exist on disk (first missing: {missing_images[0]})."
        )
        print(msg, file=sys.stderr)
        raise FileNotFoundError(msg)

    case_ids = {c["case_id"] for c in cases}
    labels_all = load_labels(labels_path)
    split_labels = {cid: labels_all[cid] for cid in case_ids if cid in labels_all}
    labelled_count = len(split_labels)
    excluded_legacy_count = sum(1 for lbls in split_labels.values() if "broken_anatomy" in lbls)

    if competitors is None:
        competitors = [
            NullCompetitor(),
            FrameJuryCompetitor(),
        ]

    rows: list[dict[str, Any]] = []

    for comp in competitors:
        per_run_means_ms: list[float] = []
        last_results: dict[str, CompetitorResult] = {}

        tracemalloc.start()
        tracemalloc.reset_peak()

        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            run_results: dict[str, CompetitorResult] = {}
            for case in cases:
                run_results[case["case_id"]] = comp.judge_case(case)
            elapsed_s = time.perf_counter() - t0
            per_case_mean_ms = (elapsed_s * 1000.0) / len(cases) if cases else 0.0
            per_run_means_ms.append(per_case_mean_ms)
            last_results = run_results

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        median_ms = statistics.median(per_run_means_ms) if per_run_means_ms else 0.0
        peak_mb = peak_bytes / (1024 * 1024)

        for defect in DEFECTS:
            s = score(last_results, split_labels, defect)
            abstain_pct = (s["abstain_rate"] * 100.0) if s["abstain_rate"] is not None else None
            rows.append(
                {
                    "system": comp.name,
                    "defect": defect,
                    "precision": s["precision"],
                    "recall": s["recall"],
                    "f1": s["f1"],
                    "abstain_pct": abstain_pct,
                    "ms_per_frame": median_ms,
                    "peak_ram_mb": peak_mb,
                    "licence": comp.licence,
                }
            )

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    pinned_versions = get_pinned_versions()

    content = render_leaderboard(
        split_name=split,
        case_count=len(cases),
        labelled_count=labelled_count,
        excluded_legacy_count=excluded_legacy_count,
        timestamp=timestamp,
        pinned_versions=pinned_versions,
        rows=rows,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return content


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("benchmarks/cases/cases.jsonl"),
        help="Path to cases JSONL file",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("benchmarks/labels/labels.jsonl"),
        help="Path to ground-truth labels JSONL file (optional)",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("benchmarks/cases/splits.json"),
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Corpus split to evaluate ('dev' or 'test', default 'test')",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/LEADERBOARD.md"),
        help="Destination path for LEADERBOARD.md",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for timing protocol (default 3)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_benchmark(
            cases_path=args.cases,
            labels_path=args.labels,
            splits_path=args.splits,
            split=args.split,
            out_path=args.out,
            runs=args.runs,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Leaderboard written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
