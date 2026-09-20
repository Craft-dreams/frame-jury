"""Strict, dependency-free schemas for corpus cases and JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

CASE_SCHEMA_VERSION = "1.0"
ENTITY_KINDS = frozenset({"character", "object", "environment"})


class CaseValidationError(ValueError):
    """Raised when a corpus case does not match the versioned schema."""


def _fail(location: str, message: str) -> None:
    raise CaseValidationError(f"{location}: {message}")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        _fail(location, f"missing field(s): {', '.join(sorted(missing))}")
    if extra:
        _fail(location, f"unknown field(s): {', '.join(sorted(extra))}")


def _string(value: Any, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _fail(location, "must be a non-empty string" if not allow_empty else "must be a string")
    return value


def _absolute_path(value: Any, location: str) -> str:
    path = _string(value, location)
    if not Path(path).is_absolute():
        _fail(location, "must be an absolute path")
    return path


def validate_case(value: Any) -> dict[str, Any]:
    """Validate one case and return it unchanged with a precise error on failure."""

    case = _mapping(value, "case")
    _exact_keys(
        case,
        {"schema_version", "case_id", "image_path", "shot", "reference_images", "provenance"},
        "case",
    )
    if case["schema_version"] != CASE_SCHEMA_VERSION:
        _fail("case.schema_version", f"must equal {CASE_SCHEMA_VERSION!r}")
    case_id = _string(case["case_id"], "case.case_id")
    image_path = _absolute_path(case["image_path"], "case.image_path")

    shot = _mapping(case["shot"], "case.shot")
    _exact_keys(
        shot,
        {"shot_id", "framing", "declared_entities", "staging", "positive_prompt", "negative_prompt"},
        "case.shot",
    )
    shot_id = _string(shot["shot_id"], "case.shot.shot_id")
    if not case_id.endswith(f"/{shot_id}"):
        _fail("case.case_id", "must end with '/' followed by shot.shot_id")
    if Path(image_path).stem != shot_id:
        _fail("case.image_path", "file stem must equal shot.shot_id")
    _string(shot["framing"], "case.shot.framing")
    _string(shot["staging"], "case.shot.staging")
    _string(shot["positive_prompt"], "case.shot.positive_prompt")
    _string(shot["negative_prompt"], "case.shot.negative_prompt", allow_empty=True)

    entities = shot["declared_entities"]
    if not isinstance(entities, list):
        _fail("case.shot.declared_entities", "must be an array")
    seen_entities: set[str] = set()
    entity_references: dict[str, list[str]] = {}
    for index, raw_entity in enumerate(entities):
        location = f"case.shot.declared_entities[{index}]"
        entity = _mapping(raw_entity, location)
        _exact_keys(entity, {"entity_id", "kind", "reference_images"}, location)
        entity_id = _string(entity["entity_id"], f"{location}.entity_id")
        if entity_id in seen_entities:
            _fail(f"{location}.entity_id", "must be unique within the declaration")
        seen_entities.add(entity_id)
        if entity["kind"] not in ENTITY_KINDS:
            _fail(f"{location}.kind", f"must be one of {', '.join(sorted(ENTITY_KINDS))}")
        references = entity["reference_images"]
        if not isinstance(references, list):
            _fail(f"{location}.reference_images", "must be an array")
        if len(set(references)) != len(references):
            _fail(f"{location}.reference_images", "must not contain duplicates")
        entity_references[entity_id] = [
            _absolute_path(path, f"{location}.reference_images[{reference_index}]")
            for reference_index, path in enumerate(references)
        ]

    references = _mapping(case["reference_images"], "case.reference_images")
    if set(references) != seen_entities:
        _fail("case.reference_images", "keys must exactly match the declared entity ids")
    for entity_id, paths in references.items():
        if not isinstance(paths, list):
            _fail(f"case.reference_images.{entity_id}", "must be an array")
        validated = [
            _absolute_path(path, f"case.reference_images.{entity_id}[{index}]")
            for index, path in enumerate(paths)
        ]
        if validated != entity_references[entity_id]:
            _fail(
                f"case.reference_images.{entity_id}",
                "must equal the reference_images on the declared entity",
            )

    provenance = _mapping(case["provenance"], "case.provenance")
    _exact_keys(provenance, {"run_id", "model", "seed"}, "case.provenance")
    run_id = _string(provenance["run_id"], "case.provenance.run_id")
    if case_id != f"{run_id}/{shot_id}":
        _fail("case.case_id", "must equal provenance.run_id + '/' + shot.shot_id")
    _string(provenance["model"], "case.provenance.model")
    seed = provenance["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        _fail("case.provenance.seed", "must be a non-negative integer")

    return dict(case)


def iter_cases(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated cases from a JSONL file, reporting its exact bad line."""

    source = Path(path)
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CaseValidationError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            try:
                case = validate_case(value)
            except CaseValidationError as exc:
                raise CaseValidationError(f"{source}:{line_number}: {exc}") from exc
            case_id = case["case_id"]
            if case_id in seen:
                raise CaseValidationError(f"{source}:{line_number}: duplicate case_id {case_id!r}")
            seen.add(case_id)
            yield case


def write_cases(path: str | Path, cases: Iterable[Mapping[str, Any]]) -> None:
    """Validate and atomically write cases as deterministic UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        seen: set[str] = set()
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                validated = validate_case(case)
                case_id = validated["case_id"]
                if case_id in seen:
                    raise CaseValidationError(f"duplicate case_id {case_id!r}")
                seen.add(case_id)
                handle.write(json.dumps(validated, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
