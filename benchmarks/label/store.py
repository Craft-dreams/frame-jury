"""Append-only label persistence and resumability."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from benchmarks.corpus.schema import iter_cases

from .schema import LabelValidationError, validate_label


class LabelStore:
    """Own validated cases and append decisions without ever truncating labels."""

    def __init__(self, cases_path: str | Path, labels_path: str | Path):
        self.cases_path = Path(cases_path)
        self.labels_path = Path(labels_path)
        self.cases = list(iter_cases(self.cases_path))
        self._by_id = {case["case_id"]: case for case in self.cases}
        self._lock = threading.Lock()
        self._labelled = self._read_labelled_ids()

    @property
    def labelled_count(self) -> int:
        return len(self._labelled)

    @property
    def total_count(self) -> int:
        return len(self.cases)

    def _read_labelled_ids(self) -> set[str]:
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
        return labelled

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
