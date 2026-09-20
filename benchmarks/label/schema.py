"""Schema for append-only human judgements."""

from __future__ import annotations

from datetime import datetime
from typing import Any

LABEL_SCHEMA_VERSION = "1.0"
DEFECTS = (
    "duplicated_character",
    "extra_person",
    "missing_entity",
    "wrong_identity",
    "broken_anatomy",
    "fused_objects",
    "garbled_text",
    "empty_or_flat",
)
DECISIONS = frozenset((*DEFECTS, "clean", "uncertain"))


class LabelValidationError(ValueError):
    """Raised when a label record is malformed."""


def validate_label(value: Any, *, known_case_ids: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelValidationError("label must be an object")
    expected = {"schema_version", "case_id", "defects", "notes", "labeller", "at"}
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        raise LabelValidationError(f"label missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise LabelValidationError(f"label has unknown field(s): {', '.join(sorted(extra))}")
    if value["schema_version"] != LABEL_SCHEMA_VERSION:
        raise LabelValidationError(f"schema_version must equal {LABEL_SCHEMA_VERSION!r}")
    case_id = value["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise LabelValidationError("case_id must be a non-empty string")
    if known_case_ids is not None and case_id not in known_case_ids:
        raise LabelValidationError(f"unknown case_id {case_id!r}")
    defects = value["defects"]
    if not isinstance(defects, list) or not defects:
        raise LabelValidationError("defects must be a non-empty array")
    if any(not isinstance(defect, str) or defect not in DECISIONS for defect in defects):
        raise LabelValidationError(f"defects may contain only: {', '.join(sorted(DECISIONS))}")
    if len(defects) != len(set(defects)):
        raise LabelValidationError("defects must not contain duplicates")
    if ("clean" in defects or "uncertain" in defects) and len(defects) != 1:
        raise LabelValidationError("clean and uncertain are exclusive decisions")
    if not isinstance(value["notes"], str):
        raise LabelValidationError("notes must be a string")
    if not isinstance(value["labeller"], str) or not value["labeller"].strip():
        raise LabelValidationError("labeller must be a non-empty string")
    at = value["at"]
    if not isinstance(at, str) or not at:
        raise LabelValidationError("at must be an ISO-8601 string")
    try:
        instant = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabelValidationError("at must be a valid ISO-8601 timestamp") from exc
    if instant.tzinfo is None:
        raise LabelValidationError("at must include a timezone")
    return dict(value)
