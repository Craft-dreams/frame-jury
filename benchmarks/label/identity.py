"""Validation and append-only persistence for per-character identity labels."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.corpus.schema import iter_cases

from .schema import LabelValidationError, validate_label
from .store import queue_order

IDENTITY_SCHEMA_VERSION = "1.0"
IDENTITY_DECISIONS = ("same", "different", "not_visible", "unsure")
_IDENTITY_FIELDS = {
    "schema_version",
    "case_id",
    "entity_id",
    "decision",
    "faces",
    "face_boxes",
    "labeller",
    "at",
}


class IdentityLabelError(ValueError):
    """Raised when an identity label record is malformed."""


def validate_identity_label(
    value: Any,
    *,
    known_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate an identity record and return a shallow copy."""

    if not isinstance(value, dict):
        raise IdentityLabelError("identity label must be an object")
    missing = _IDENTITY_FIELDS - value.keys()
    extra = value.keys() - _IDENTITY_FIELDS
    if missing:
        raise IdentityLabelError(
            f"identity label missing field(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise IdentityLabelError(
            f"identity label has unknown field(s): {', '.join(sorted(extra))}"
        )
    if value["schema_version"] != IDENTITY_SCHEMA_VERSION:
        raise IdentityLabelError(
            f"schema_version must equal {IDENTITY_SCHEMA_VERSION!r}"
        )

    case_id = value["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise IdentityLabelError("case_id must be a non-empty string")
    entity_id = value["entity_id"]
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise IdentityLabelError("entity_id must be a non-empty string")
    if known_pairs is not None and (case_id, entity_id) not in known_pairs:
        raise IdentityLabelError(
            f"unknown case_id/entity_id pair {(case_id, entity_id)!r}"
        )

    decision = value["decision"]
    if decision not in IDENTITY_DECISIONS:
        raise IdentityLabelError(
            f"decision must be one of: {', '.join(IDENTITY_DECISIONS)}"
        )

    boxes = value["face_boxes"]
    if not isinstance(boxes, list):
        raise IdentityLabelError("face_boxes must be an array")
    for index, box in enumerate(boxes):
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                isinstance(coordinate, bool) or not isinstance(coordinate, int)
                for coordinate in box
            )
        ):
            raise IdentityLabelError(
                f"face_boxes[{index}] must be an array of four integers"
            )

    faces = value["faces"]
    if not isinstance(faces, list):
        raise IdentityLabelError("faces must be an array")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in faces):
        raise IdentityLabelError("faces must contain only integer indices")
    if len(faces) != len(set(faces)):
        raise IdentityLabelError("faces must not contain duplicate indices")
    if any(index < 0 or index >= len(boxes) for index in faces):
        raise IdentityLabelError("faces contains an index outside face_boxes")
    if decision == "same" and not faces:
        raise IdentityLabelError("faces must contain at least one index when decision is 'same'")
    if decision in {"not_visible", "unsure"} and faces:
        raise IdentityLabelError(f"faces must be empty when decision is {decision!r}")

    if not isinstance(value["labeller"], str) or not value["labeller"].strip():
        raise IdentityLabelError("labeller must be a non-empty string")
    at = value["at"]
    if not isinstance(at, str) or not at:
        raise IdentityLabelError("at must be an ISO-8601 string")
    try:
        instant = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IdentityLabelError("at must be a valid ISO-8601 timestamp") from exc
    if instant.tzinfo is None:
        raise IdentityLabelError("at must include a timezone")
    return dict(value)


def identity_pairs(case: dict[str, Any]) -> list[str]:
    """Return reference-backed declared characters in declaration order."""

    references = case["reference_images"]
    return [
        entity["entity_id"]
        for entity in case["shot"]["declared_entities"]
        if entity["kind"] == "character" and references.get(entity["entity_id"], [])
    ]


class IdentityStore:
    """Own identity pairs and append decisions without rewriting prior records."""

    def __init__(
        self,
        cases_path: str | Path,
        identity_labels_path: str | Path,
        frame_labels_path: str | Path | None = None,
    ):
        self.cases_path = Path(cases_path)
        self.identity_labels_path = Path(identity_labels_path)
        self.frame_labels_path = Path(frame_labels_path) if frame_labels_path is not None else None
        cases = list(iter_cases(self.cases_path))
        self._by_id = {case["case_id"]: case for case in cases}
        frame_labelled = self._read_frame_labelled()
        self.cases = sorted(
            cases,
            key=lambda case: (case["case_id"] not in frame_labelled, queue_order(case)),
        )
        self._pairs = [
            (case["case_id"], entity_id)
            for case in self.cases
            for entity_id in identity_pairs(case)
        ]
        self._known_pairs = set(self._pairs)
        self._lock = threading.Lock()
        self._labels = self._read_identity_labels()

    @property
    def labelled_count(self) -> int:
        return len(self._labels)

    @property
    def total_count(self) -> int:
        return len(self._pairs)

    def _read_frame_labelled(self) -> set[str]:
        labelled: set[str] = set()
        if self.frame_labels_path is None or not self.frame_labels_path.exists():
            return labelled
        known_case_ids = set(self._by_id)
        with self.frame_labels_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    label = validate_label(value, known_case_ids=known_case_ids)
                except (json.JSONDecodeError, LabelValidationError) as exc:
                    raise IdentityLabelError(
                        f"{self.frame_labels_path}:{line_number}: {exc}"
                    ) from exc
                labelled.add(label["case_id"])
        return labelled

    def _read_identity_labels(self) -> dict[tuple[str, str], dict[str, Any]]:
        labels: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.identity_labels_path.exists():
            return labels
        with self.identity_labels_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IdentityLabelError(
                        f"{self.identity_labels_path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                try:
                    label = validate_identity_label(value, known_pairs=self._known_pairs)
                except IdentityLabelError as exc:
                    raise IdentityLabelError(
                        f"{self.identity_labels_path}:{line_number}: {exc}"
                    ) from exc
                labels[(label["case_id"], label["entity_id"])] = label
        return labels

    def next_pair(self) -> tuple[dict[str, Any], str] | None:
        for case in self.cases:
            for entity_id in identity_pairs(case):
                if (case["case_id"], entity_id) not in self._labels:
                    return case, entity_id
        return None

    def case(self, case_id: str) -> dict[str, Any] | None:
        return self._by_id.get(case_id)

    def append(self, record: dict[str, Any]) -> None:
        validated = validate_identity_label(record, known_pairs=self._known_pairs)
        pair = (validated["case_id"], validated["entity_id"])
        with self._lock:
            self.identity_labels_path.parent.mkdir(parents=True, exist_ok=True)
            with self.identity_labels_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(validated, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._labels[pair] = validated
