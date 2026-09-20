"""Join read-only Content Factory run artifacts into corpus cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import CASE_SCHEMA_VERSION, CaseValidationError, validate_case, write_cases

_STAGES = {
    "missing_direction": Path("07-direction") / "outcome.json",
    "missing_world": Path("03-world") / "canonical-world.json",
    "missing_reference_sheet": Path("05-production-bible") / "reference-sheet",
    "missing_visuals": Path("10-resolved-media") / "visuals",
}


class SourceDataError(ValueError):
    """A source artifact cannot produce a valid corpus case."""


@dataclass
class BuildReport:
    runs_seen: int = 0
    runs_usable: int = 0
    cases_emitted: int = 0
    runs_skipped_by_reason: Counter[str] = field(default_factory=Counter)
    cases_skipped_by_reason: Counter[str] = field(default_factory=Counter)

    def format(self) -> str:
        lines = [
            "Corpus build summary",
            f"  runs seen: {self.runs_seen}",
            f"  runs usable: {self.runs_usable}",
            f"  cases emitted: {self.cases_emitted}",
            "  runs skipped by reason (stage reasons may overlap):",
        ]
        if self.runs_skipped_by_reason:
            lines.extend(
                f"    {reason}: {count}"
                for reason, count in sorted(self.runs_skipped_by_reason.items())
            )
        else:
            lines.append("    none: 0")
        lines.append("  cases skipped by reason:")
        if self.cases_skipped_by_reason:
            lines.extend(
                f"    {reason}: {count}"
                for reason, count in sorted(self.cases_skipped_by_reason.items())
            )
        else:
            lines.append("    none: 0")
        return "\n".join(lines)


def _read_json(path: Path, artifact: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceDataError(f"invalid_{artifact}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceDataError(f"invalid_{artifact}: root must be an object")
    return value


def _string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
    reason: str = "malformed_declaration",
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise SourceDataError(f"{reason}: {field_name} must be {requirement}")
    return value


def _declarations(direction: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    beats = direction.get("beats")
    if not isinstance(beats, list):
        raise SourceDataError("malformed_direction: beats must be an array")
    declarations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for beat_index, beat in enumerate(beats):
        if not isinstance(beat, dict) or not isinstance(beat.get("scene_plan"), dict):
            raise SourceDataError(f"malformed_direction: beats[{beat_index}].scene_plan must be an object")
        plan = beat["scene_plan"]
        scene = plan.get("scene_spec")
        shots = plan.get("shot_specs")
        if not isinstance(scene, dict) or not isinstance(shots, list):
            raise SourceDataError(
                f"malformed_direction: beats[{beat_index}].scene_plan needs scene_spec and shot_specs"
            )
        scene_id = _string(
            scene.get("scene_id"),
            f"beats[{beat_index}].scene_plan.scene_spec.scene_id",
            reason="malformed_direction",
        )
        for shot_index, shot in enumerate(shots):
            if not isinstance(shot, dict):
                raise SourceDataError(
                    f"malformed_direction: beats[{beat_index}].scene_plan.shot_specs[{shot_index}] must be an object"
                )
            shot_id = _string(
                shot.get("shot_id"), "shot_specs[].shot_id", reason="malformed_direction"
            )
            if shot_id in declarations:
                raise SourceDataError(f"malformed_direction: duplicate shot_id {shot_id!r}")
            if shot.get("scene_id") != scene_id:
                raise SourceDataError(
                    f"malformed_direction: shot {shot_id!r} does not belong to its scene_spec"
                )
            declarations[shot_id] = (shot, scene)
    return declarations


def _entity_kinds(world: dict[str, Any]) -> dict[str, str]:
    identities = world.get("identities")
    if not isinstance(identities, list):
        raise SourceDataError("malformed_world: identities must be an array")
    kinds: dict[str, str] = {}
    for index, identity in enumerate(identities):
        if not isinstance(identity, dict):
            raise SourceDataError(f"malformed_world: identities[{index}] must be an object")
        entity_id = _string(
            identity.get("entity_id"),
            f"identities[{index}].entity_id",
            reason="malformed_world",
        )
        kind = _string(
            identity.get("kind"), f"identities[{index}].kind", reason="malformed_world"
        )
        if entity_id in kinds:
            raise SourceDataError(f"malformed_world: duplicate entity_id {entity_id!r}")
        kinds[entity_id] = kind
    return kinds


def _reference_index(reference_dir: Path) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}
    for path in sorted(reference_dir.glob("*.png"), key=lambda item: item.name.casefold()):
        references.setdefault(path.stem, []).append(str(path.resolve()))
    return references


def _case_for_image(
    run: Path,
    image: Path,
    declarations: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    kinds: dict[str, str],
    references: dict[str, list[str]],
) -> dict[str, Any]:
    shot_id = image.stem
    if shot_id not in declarations:
        raise SourceDataError(f"missing_declaration: no shot_spec for {shot_id!r}")
    shot_spec, _scene = declarations[shot_id]
    camera = shot_spec.get("camera")
    if not isinstance(camera, dict):
        raise SourceDataError(f"malformed_declaration: shot {shot_id!r} camera must be an object")
    framing = _string(camera.get("framing"), f"shot {shot_id!r} camera.framing")
    staging = _string(shot_spec.get("purpose"), f"shot {shot_id!r} purpose")
    entity_ids = shot_spec.get("required_visible_entity_ids")
    if not isinstance(entity_ids, list) or any(not isinstance(item, str) or not item for item in entity_ids):
        raise SourceDataError(
            f"malformed_declaration: shot {shot_id!r} required_visible_entity_ids must be an array of strings"
        )
    if len(set(entity_ids)) != len(entity_ids):
        raise SourceDataError(
            f"malformed_declaration: shot {shot_id!r} required_visible_entity_ids contains duplicates"
        )

    declared_entities = []
    reference_map: dict[str, list[str]] = {}
    for entity_id in entity_ids:
        if entity_id not in kinds:
            raise SourceDataError(
                f"unknown_entity: shot {shot_id!r} references {entity_id!r}, absent from canonical world"
            )
        entity_references = references.get(entity_id, [])
        reference_map[entity_id] = entity_references
        declared_entities.append(
            {
                "entity_id": entity_id,
                "kind": kinds[entity_id],
                "reference_images": entity_references,
            }
        )

    sidecar_path = image.with_name(f"{shot_id}.visual.json")
    if not sidecar_path.is_file():
        raise SourceDataError(f"missing_sidecar: {sidecar_path.name}")
    sidecar = _read_json(sidecar_path, "sidecar")
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, dict):
        raise SourceDataError(f"malformed_sidecar: {sidecar_path.name} metadata must be an object")
    source_shot_id = metadata.get("source_shot_id")
    if source_shot_id is not None and source_shot_id != shot_id:
        raise SourceDataError(
            f"malformed_sidecar: {sidecar_path.name} declares source_shot_id {source_shot_id!r}"
        )
    positive_prompt = _string(
        metadata.get("prompt"),
        f"{sidecar_path.name} metadata.prompt",
        reason="malformed_sidecar",
    )
    if "negative_prompt" not in metadata:
        raise SourceDataError(
            f"malformed_sidecar: {sidecar_path.name} metadata.negative_prompt is missing"
        )
    negative_prompt = _string(
        metadata["negative_prompt"],
        f"{sidecar_path.name} metadata.negative_prompt",
        allow_empty=True,
        reason="malformed_sidecar",
    )
    model = _string(
        sidecar.get("model"), f"{sidecar_path.name} model", reason="malformed_sidecar"
    )
    seed = sidecar.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SourceDataError(f"malformed_sidecar: {sidecar_path.name} seed must be a non-negative integer")

    case = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": f"{run.name}/{shot_id}",
        "image_path": str(image.resolve()),
        "shot": {
            "shot_id": shot_id,
            "framing": framing,
            "declared_entities": declared_entities,
            "staging": staging,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
        },
        "reference_images": reference_map,
        "provenance": {"run_id": run.name, "model": model, "seed": seed},
    }
    try:
        return validate_case(case)
    except CaseValidationError as exc:
        raise SourceDataError(f"invalid_case: {exc}") from exc


def build_corpus(runs_path: str | Path) -> tuple[list[dict[str, Any]], BuildReport]:
    """Build cases without modifying anything beneath ``runs_path``."""

    root = Path(runs_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"runs directory does not exist: {root}")
    report = BuildReport()
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()

    for run in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name):
        report.runs_seen += 1
        missing = []
        for reason, relative in _STAGES.items():
            target = run / relative
            expected = target.is_dir() if reason in {"missing_reference_sheet", "missing_visuals"} else target.is_file()
            if not expected:
                missing.append(reason)
        if missing:
            # Stage preflight happens before reading any frame, so incomplete runs are atomic skips.
            for reason in missing:
                report.runs_skipped_by_reason[reason] += 1
            visual_dir = run / _STAGES["missing_visuals"]
            if visual_dir.is_dir():
                skipped_images = sum(1 for path in visual_dir.glob("*.png") if path.is_file())
                report.cases_skipped_by_reason[missing[0]] += skipped_images
            continue

        try:
            direction = _read_json(run / _STAGES["missing_direction"], "direction")
            world = _read_json(run / _STAGES["missing_world"], "world")
            declarations = _declarations(direction)
            kinds = _entity_kinds(world)
        except SourceDataError as exc:
            reason = str(exc).split(":", 1)[0]
            report.runs_skipped_by_reason[reason] += 1
            visual_dir = run / _STAGES["missing_visuals"]
            report.cases_skipped_by_reason[reason] += sum(
                1 for path in visual_dir.glob("*.png") if path.is_file()
            )
            continue

        report.runs_usable += 1
        reference_dir = run / _STAGES["missing_reference_sheet"]
        visual_dir = run / _STAGES["missing_visuals"]
        references = _reference_index(reference_dir)
        for image in sorted(visual_dir.glob("*.png"), key=lambda item: item.name.casefold()):
            try:
                case = _case_for_image(run, image, declarations, kinds, references)
            except SourceDataError as exc:
                reason = str(exc).split(":", 1)[0]
                report.cases_skipped_by_reason[reason] += 1
                continue
            if case["case_id"] in case_ids:
                report.cases_skipped_by_reason["duplicate_case_id"] += 1
                continue
            case_ids.add(case["case_id"])
            cases.append(case)

    cases.sort(key=lambda case: case["case_id"])
    report.cases_emitted = len(cases)
    return cases, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, help="Content Factory build/runs directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/cases/cases.jsonl"),
        help="destination cases.jsonl (default: benchmarks/cases/cases.jsonl)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cases, report = build_corpus(args.runs)
        write_cases(args.out, cases)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report.format())
    print(f"  output: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
