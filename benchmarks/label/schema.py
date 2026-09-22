"""Schema for append-only human judgements."""

from __future__ import annotations

from datetime import datetime
from typing import Any

LABEL_SCHEMA_VERSION = "2.0"
TAXONOMY_VERSION = "2.1"
LEGACY_TAXONOMY_VERSION = "2.0"
LEGACY_LABEL_SCHEMA_VERSION = "1.0"
DEFECTS_V2_0 = (
    "duplicated_character",
    "broken_hands",
    "extra_person",
    "missing_entity",
    "wrong_identity",
    "broken_body",
    "wrong_scale",
    "fused_objects",
    "garbled_text",
    "empty_or_flat",
)
DECISIONS_V2_0 = frozenset((*DEFECTS_V2_0, "clean", "uncertain"))

DEFECTS = (
    *DEFECTS_V2_0,
    "broken_face",
    "wrong_interaction",
)
DECISIONS = frozenset((*DEFECTS, "clean", "uncertain"))

DEFECT_DESCRIPTIONS_PT: dict[str, str] = {
    "duplicated_character": "O mesmo personagem aparece mais de uma vez",
    "broken_hands": "Mãos ou dedos deformados, fundidos ou em número errado",
    "extra_person": "Há gente onde a cena diz que não há mais ninguém",
    "missing_entity": "Um personagem ou objeto declarado não aparece",
    "wrong_identity": "O personagem não parece a pessoa da ficha de referência",
    "broken_body": "Corpo, membros, articulações ou pose impossíveis",
    "wrong_scale": "Tamanho ou proporção errada em relação à cena ou aos outros",
    "fused_objects": "Dois objetos ou entidades se fundiram num só",
    "garbled_text": "Texto ilegível ou que não forma palavras",
    "empty_or_flat": "Quadro vazio, chapado ou sem detalhe",
    "broken_face": "Rosto deformado, derretido ou com feições fora do lugar",
    "wrong_interaction": (
        "Objeto segurado ou usado de forma errada ou impossível "
        "(pegada errada, flutuando, atravessando a mão)"
    ),
    "clean": "Nada errado",
    "uncertain": "Não tenho certeza",
}
LEGACY_DEFECTS = (
    "duplicated_character",
    "extra_person",
    "missing_entity",
    "wrong_identity",
    "broken_anatomy",
    "fused_objects",
    "garbled_text",
    "empty_or_flat",
)
LEGACY_DECISIONS = frozenset((*LEGACY_DEFECTS, "clean", "uncertain"))


class LabelValidationError(ValueError):
    """Raised when a label record is malformed."""


def validate_label(value: Any, *, known_case_ids: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelValidationError("label must be an object")
    schema_version = value.get("schema_version")
    if schema_version == LABEL_SCHEMA_VERSION:
        expected = {
            "schema_version",
            "taxonomy_version",
            "case_id",
            "defects",
            "notes",
            "labeller",
            "at",
        }
    elif schema_version == LEGACY_LABEL_SCHEMA_VERSION:
        expected = {"schema_version", "case_id", "defects", "notes", "labeller", "at"}
        decisions = LEGACY_DECISIONS
    else:
        raise LabelValidationError(
            "schema_version must equal "
            f"{LABEL_SCHEMA_VERSION!r} (current) or {LEGACY_LABEL_SCHEMA_VERSION!r} (legacy)"
        )
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        raise LabelValidationError(f"label missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise LabelValidationError(f"label has unknown field(s): {', '.join(sorted(extra))}")
    if schema_version == LABEL_SCHEMA_VERSION:
        taxonomy_version = value["taxonomy_version"]
        if taxonomy_version == TAXONOMY_VERSION:
            decisions = DECISIONS
        elif taxonomy_version == LEGACY_TAXONOMY_VERSION:
            decisions = DECISIONS_V2_0
        else:
            raise LabelValidationError(
                f"taxonomy_version must equal {TAXONOMY_VERSION!r} or {LEGACY_TAXONOMY_VERSION!r}"
            )
    case_id = value["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise LabelValidationError("case_id must be a non-empty string")
    if known_case_ids is not None and case_id not in known_case_ids:
        raise LabelValidationError(f"unknown case_id {case_id!r}")
    defects = value["defects"]
    if not isinstance(defects, list) or not defects:
        raise LabelValidationError("defects must be a non-empty array")
    if any(not isinstance(defect, str) or defect not in decisions for defect in defects):
        raise LabelValidationError(f"defects may contain only: {', '.join(sorted(decisions))}")
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
