"""Append-only label persistence and resumability."""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.corpus.schema import iter_cases

from .schema import DEFECTS, LabelValidationError, validate_label


def _framing_band(framing: str) -> int:
    normalized = framing.casefold()
    if "close" in normalized or "medium" in normalized:
        return 0
    if "wide" in normalized:
        return 1
    return 2


def queue_order(case: dict[str, Any]) -> tuple[int, int, str, str]:
    """Prioritize likely defects while keeping runs together inside each band."""

    character_count = sum(
        entity["kind"] == "character" for entity in case["shot"]["declared_entities"]
    )
    character_band = 0 if character_count >= 2 else 1 if character_count == 1 else 2
    return (
        character_band,
        _framing_band(case["shot"]["framing"]),
        case["provenance"]["run_id"].casefold(),
        case["case_id"].casefold(),
    )


class LabelStore:
    """Own validated cases and append decisions without ever truncating labels."""

    def __init__(self, cases_path: str | Path, labels_path: str | Path):
        self.cases_path = Path(cases_path)
        self.labels_path = Path(labels_path)
        self.cases = sorted(iter_cases(self.cases_path), key=queue_order)
        self._by_id = {case["case_id"]: case for case in self.cases}
        self._lock = threading.Lock()
        self._positive_counts: Counter[str] = Counter({defect: 0 for defect in DEFECTS})
        self._legacy_broken_anatomy_count = 0
        self._labelled = self._read_labels()

    @property
    def labelled_count(self) -> int:
        return len(self._labelled)

    @property
    def total_count(self) -> int:
        return len(self.cases)

    @property
    def positive_counts(self) -> dict[str, int]:
        return {defect: self._positive_counts[defect] for defect in DEFECTS}

    @property
    def legacy_broken_anatomy_count(self) -> int:
        return self._legacy_broken_anatomy_count

    def _read_labels(self) -> set[str]:
        labelled: set[str] = set()
        if not self.labels_path.exists():
            return labelled
        with self.labels_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LabelValidationError(
                        f"{self.labels_path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                try:
                    label = validate_label(value, known_case_ids=set(self._by_id))
                except LabelValidationError as exc:
                    raise LabelValidationError(f"{self.labels_path}:{line_number}: {exc}") from exc
                case_id = label["case_id"]
                if case_id in labelled:
                    raise LabelValidationError(
                        f"{self.labels_path}:{line_number}: duplicate label for {case_id!r}"
                    )
                labelled.add(case_id)
                self._record_counts(label["defects"])
        return labelled

    def _record_counts(self, defects: list[str]) -> None:
        for defect in defects:
            if defect in self._positive_counts:
                self._positive_counts[defect] += 1
            elif defect == "broken_anatomy":
                self._legacy_broken_anatomy_count += 1

    def next_case(self) -> dict[str, Any] | None:
        return next((case for case in self.cases if case["case_id"] not in self._labelled), None)

    def case(self, case_id: str) -> dict[str, Any] | None:
        return self._by_id.get(case_id)

    def append(self, label: dict[str, Any]) -> None:
        validated = validate_label(label, known_case_ids=set(self._by_id))
        case_id = validated["case_id"]
        with self._lock:
            if case_id in self._labelled:
                raise LabelValidationError(f"case {case_id!r} is already labelled")
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            with self.labels_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(validated, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._labelled.add(case_id)
            self._record_counts(validated["defects"])
