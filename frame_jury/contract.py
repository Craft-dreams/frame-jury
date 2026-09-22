"""frame_jury.contract — request and verdict dataclasses, JSON in and out.

This is the entire public surface between frame-jury and its callers.
The factory's adapter speaks exactly this schema; nothing else is imported
across the boundary.  See SPEC.md §4 for the canonical JSON examples.

Schema version: 2.0
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ──────────────────────────────────────────────────────────
# Schema version — bumped here, checked on every deserialise.
# ──────────────────────────────────────────────────────────
SCHEMA_VERSION = "2.0"

# ──────────────────────────────────────────────────────────
# Taxonomy constants  (SPEC.md §5)
# ──────────────────────────────────────────────────────────
VALID_CHECKS: frozenset[str] = frozenset(
    {"presence", "identity", "anatomy", "legibility", "vlm_scene"}
)
VALID_BUDGETS: frozenset[str] = frozenset({"cheap", "full"})
VALID_VERDICTS: frozenset[str] = frozenset({"accept", "reject", "unsure"})
VALID_SEVERITIES: frozenset[str] = frozenset({"blocking", "warning", "info"})
VALID_ENTITY_KINDS: frozenset[str] = frozenset({"character", "object", "environment"})

# Presence defects (SPEC.md §5)
DEFECT_DUPLICATED_CHARACTER = "duplicated_character"
DEFECT_EXTRA_PERSON = "extra_person"
DEFECT_MISSING_ENTITY = "missing_entity"

# Identity defects (SPEC.md §5)
DEFECT_WRONG_IDENTITY = "wrong_identity"

# Anatomy defects (SPEC.md §5)
DEFECT_BROKEN_FACE = "broken_face"

# Interaction defects (SPEC.md §5)
DEFECT_WRONG_INTERACTION = "wrong_interaction"


# ──────────────────────────────────────────────────────────
# Request side
# ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entity:
    """One declared entity inside a shot."""

    entity_id: str
    kind: Literal["character", "object", "environment"]
    display_name: str
    aliases: tuple[str, ...]
    reference_images: tuple[str, ...]
    # Optional profile fields — present together or not at all (SPEC.md §7).
    visual_identity: str = ""
    relative_scale: str = ""
    approximate_dimensions: str = ""
    is_collective: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any], location: str = "entity") -> "Entity":
        _require(d, "entity_id", location)
        _require(d, "kind", location)
        _require(d, "display_name", location)
        kind = d["kind"]
        if kind not in VALID_ENTITY_KINDS:
            raise ContractError(
                f"{location}.kind: must be one of {sorted(VALID_ENTITY_KINDS)}, got {kind!r}"
            )
        return cls(
            entity_id=_str(d, "entity_id", location),
            kind=kind,  # type: ignore[arg-type]
            display_name=_str(d, "display_name", location),
            aliases=tuple(d.get("aliases", [])),
            reference_images=tuple(d.get("reference_images", [])),
            visual_identity=d.get("visual_identity", ""),
            relative_scale=d.get("relative_scale", ""),
            approximate_dimensions=d.get("approximate_dimensions", ""),
            is_collective=_bool(d, "is_collective", location, default=False),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "reference_images": list(self.reference_images),
        }
        if self.visual_identity:
            d["visual_identity"] = self.visual_identity
            d["relative_scale"] = self.relative_scale
            d["approximate_dimensions"] = self.approximate_dimensions
        if self.is_collective:
            d["is_collective"] = True
        return d


@dataclass(frozen=True)
class Staging:
    """Shot staging prose from the direction."""

    purpose: str
    must_render: tuple[str, ...]
    composition: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any], location: str = "staging") -> "Staging":
        return cls(
            purpose=_str(d, "purpose", location),
            must_render=tuple(d.get("must_render", [])),
            composition=tuple(d.get("composition", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "must_render": list(self.must_render),
            "composition": list(self.composition),
        }


@dataclass(frozen=True)
class Shot:
    """The declaration for one shot."""

    shot_id: str
    framing: str
    declared_entities: tuple[Entity, ...]
    staging: Staging
    positive_prompt: str
    negative_prompt: str
    other_people_allowed: bool = True
    expected_people_count: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], location: str = "shot") -> "Shot":
        _require(d, "shot_id", location)
        _require(d, "framing", location)
        raw_entities = d.get("declared_entities", [])
        entities = tuple(
            Entity.from_dict(e, f"{location}.declared_entities[{i}]")
            for i, e in enumerate(raw_entities)
        )
        return cls(
            shot_id=_str(d, "shot_id", location),
            framing=_str(d, "framing", location),
            declared_entities=entities,
            staging=Staging.from_dict(d.get("staging", {}), f"{location}.staging"),
            positive_prompt=d.get("positive_prompt", ""),
            negative_prompt=d.get("negative_prompt", ""),
            other_people_allowed=_bool(d, "other_people_allowed", location, default=True),
            expected_people_count=_optional_nonnegative_int(
                d, "expected_people_count", location
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "shot_id": self.shot_id,
            "framing": self.framing,
            "declared_entities": [e.to_dict() for e in self.declared_entities],
            "staging": self.staging.to_dict(),
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "other_people_allowed": self.other_people_allowed,
        }
        if self.expected_people_count is not None:
            d["expected_people_count"] = self.expected_people_count
        return d

    # Convenience helpers used by the checks.

    @property
    def declared_character_count(self) -> int:
        """Number of entities whose kind is 'character'."""
        return sum(1 for e in self.declared_entities if e.kind == "character")

    @property
    def declared_object_count(self) -> int:
        """Number of entities whose kind is 'object'."""
        return sum(1 for e in self.declared_entities if e.kind == "object")

    def characters(self) -> list[Entity]:
        return [e for e in self.declared_entities if e.kind == "character"]

    def non_collective_characters(self) -> list[Entity]:
        return [e for e in self.characters() if not e.is_collective]

    def objects(self) -> list[Entity]:
        return [e for e in self.declared_entities if e.kind == "object"]


@dataclass(frozen=True)
class JuryRequest:
    """One call, one frame — the full request the caller sends.

    Deserialise with :meth:`from_json` or :meth:`from_dict`.
    The factory's adapter must produce exactly this schema.
    """

    schema_version: str
    image_path: str
    shot: Shot
    checks: tuple[str, ...]  # subset of VALID_CHECKS; all checks if empty
    budget: Literal["cheap", "full"]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JuryRequest":
        sv = d.get("schema_version", "")
        if sv != SCHEMA_VERSION:
            raise ContractError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {sv!r}"
            )
        image_path = _str(d, "image_path", "request")
        if not image_path:
            raise ContractError("request.image_path: must not be empty")
        raw_checks = d.get("checks", list(VALID_CHECKS))
        for c in raw_checks:
            if c not in VALID_CHECKS:
                raise ContractError(
                    f"request.checks: unknown check {c!r}; valid: {sorted(VALID_CHECKS)}"
                )
        budget = d.get("budget", "cheap")
        if budget not in VALID_BUDGETS:
            raise ContractError(
                f"request.budget: must be one of {sorted(VALID_BUDGETS)}, got {budget!r}"
            )
        return cls(
            schema_version=sv,
            image_path=image_path,
            shot=Shot.from_dict(d.get("shot", {})),
            checks=tuple(raw_checks),
            budget=budget,  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, text: str) -> "JuryRequest":
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON in request: {exc}") from exc
        return cls.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_path": self.image_path,
            "shot": self.shot.to_dict(),
            "checks": list(self.checks),
            "budget": self.budget,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ──────────────────────────────────────────────────────────
# Verdict side
# ──────────────────────────────────────────────────────────


@dataclass
class Finding:
    """One defect instance with the evidence that produced it.

    Every finding carries its evidence (SPEC.md §4): the numbers that produced
    it, not only a label.  ``prompt_hint`` is a suggestion for a human or a
    station — frame-jury never rewrites a prompt (SPEC.md §4).
    """

    check: str                         # which check produced this
    defect: str                        # defect name from the taxonomy (SPEC.md §5)
    severity: Literal["blocking", "warning", "info"]
    confidence: float                  # [0.0, 1.0]
    evidence: dict[str, Any]           # the numbers: counts, boxes, distances …
    explanation: str                   # human-readable sentence
    entity_id: str = ""               # the entity involved, if any
    prompt_hint: str = ""             # never an edit; suggestion only

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "check": self.check,
            "defect": self.defect,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "explanation": self.explanation,
        }
        if self.entity_id:
            d["entity_id"] = self.entity_id
        if self.prompt_hint:
            d["prompt_hint"] = self.prompt_hint
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Finding":
        return cls(
            check=d["check"],
            defect=d["defect"],
            severity=d["severity"],
            confidence=float(d["confidence"]),
            evidence=d.get("evidence", {}),
            explanation=d.get("explanation", ""),
            entity_id=d.get("entity_id", ""),
            prompt_hint=d.get("prompt_hint", ""),
        )


@dataclass
class Abstention:
    """A distinct, first-class record of a check abstaining (SPEC.md §4).

    An abstention is not a finding: a defect is something wrong with the frame,
    an abstention is something the judge could not determine.
    """

    check: str            # which check abstained
    reason: str           # a stable machine-readable slug, e.g. "no_face_in_frame"
    entity_id: str = ""   # the entity it could not judge, if any
    explanation: str = "" # one human-readable sentence

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "check": self.check,
            "reason": self.reason,
        }
        if self.entity_id:
            d["entity_id"] = self.entity_id
        if self.explanation:
            d["explanation"] = self.explanation
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Abstention":
        return cls(
            check=d["check"],
            reason=d["reason"],
            entity_id=d.get("entity_id", ""),
            explanation=d.get("explanation", ""),
        )


@dataclass
class Verdict:
    """The result of judging one frame.

    ``unsure`` is a first-class answer (SPEC.md §4): a judge that guesses is
    worse than one that abstains, because the factory would learn to ignore it.

    Serialise with :meth:`to_json`; deserialise with :meth:`from_json`.
    """

    schema_version: str
    shot_id: str
    verdict: Literal["accept", "reject", "unsure"]
    confidence: float                       # aggregate confidence in the verdict
    findings: list[Finding] = field(default_factory=list)
    abstentions: list[Abstention] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    detectors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shot_id": self.shot_id,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "findings": [f.to_dict() for f in self.findings],
            "abstentions": [a.to_dict() for a in self.abstentions],
            "measurements": self.measurements,
            "timings_ms": self.timings_ms,
            "detectors": self.detectors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Verdict":
        sv = d.get("schema_version", "")
        if sv != SCHEMA_VERSION:
            raise ContractError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {sv!r}"
            )
        v = d["verdict"]
        if v not in VALID_VERDICTS:
            raise ContractError(f"verdict must be one of {sorted(VALID_VERDICTS)}, got {v!r}")
        return cls(
            schema_version=sv,
            shot_id=d["shot_id"],
            verdict=v,  # type: ignore[arg-type]
            confidence=float(d.get("confidence", 0.0)),
            findings=[Finding.from_dict(f) for f in d.get("findings", [])],
            abstentions=[Abstention.from_dict(a) for a in d.get("abstentions", [])],
            measurements=d.get("measurements", {}),
            timings_ms=d.get("timings_ms", {}),
            detectors=d.get("detectors", []),
        )

    @classmethod
    def from_json(cls, text: str) -> "Verdict":
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON in verdict: {exc}") from exc
        return cls.from_dict(d)


# ──────────────────────────────────────────────────────────
# Builder for Verdict (used by jury.py and checks)
# ──────────────────────────────────────────────────────────


class VerdictBuilder:
    """Accumulates findings and abstentions from multiple checks and emits a Verdict.

    The final verdict:
    - is ``reject`` when any finding has severity ``blocking``;
    - is ``unsure`` when any check emits an abstention and none block;
    - is ``accept`` otherwise.  A ``warning`` finding on its own produces
      ``accept`` with warnings attached (SPEC.md §4).

    Confidence is the minimum confidence across all findings (conservative: the
    verdict is only as strong as its weakest piece of evidence).
    """

    def __init__(self, shot_id: str) -> None:
        self._shot_id = shot_id
        self._findings: list[Finding] = []
        self._abstentions: list[Abstention] = []
        self._measurements: dict[str, Any] = {}
        self._timings_ms: dict[str, float] = {}
        self._detectors: list[dict[str, Any]] = []
        self._started = time.perf_counter()

    def add_finding(self, finding: Finding) -> None:
        self._findings.append(finding)

    def add_findings(self, findings: list[Finding]) -> None:
        self._findings.extend(findings)

    def add_abstention(self, abstention: Abstention) -> None:
        self._abstentions.append(abstention)

    def add_abstentions(self, abstentions: list[Abstention]) -> None:
        self._abstentions.extend(abstentions)

    def update_measurements(self, measurements: dict[str, Any]) -> None:
        self._measurements.update(measurements)

    def record_timing(self, check: str, elapsed_ms: float) -> None:
        self._timings_ms[check] = round(elapsed_ms, 1)

    def record_detector(self, check: str, backend: str, weights_sha256: str) -> None:
        self._detectors.append(
            {"check": check, "backend": backend, "weights_sha256": weights_sha256}
        )

    def build(self) -> Verdict:
        if any(f.severity == "blocking" for f in self._findings):
            verdict: Literal["accept", "reject", "unsure"] = "reject"
        elif self._abstentions:
            verdict = "unsure"
        else:
            verdict = "accept"

        confidences = [f.confidence for f in self._findings]
        confidence = min(confidences) if confidences else 1.0

        return Verdict(
            schema_version=SCHEMA_VERSION,
            shot_id=self._shot_id,
            verdict=verdict,
            confidence=confidence,
            findings=list(self._findings),
            abstentions=list(self._abstentions),
            measurements=dict(self._measurements),
            timings_ms=dict(self._timings_ms),
            detectors=list(self._detectors),
        )


# ──────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────


class ContractError(ValueError):
    """Raised when a request or verdict does not match the versioned schema."""


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────


def _require(d: dict[str, Any], key: str, location: str) -> None:
    if key not in d:
        raise ContractError(f"{location}.{key}: required field missing")


def _str(d: dict[str, Any], key: str, location: str) -> str:
    v = d.get(key, "")
    if not isinstance(v, str):
        raise ContractError(f"{location}.{key}: must be a string")
    return v


def _bool(d: dict[str, Any], key: str, location: str, default: bool = True) -> bool:
    v = d.get(key, default)
    if not isinstance(v, bool):
        raise ContractError(f"{location}.{key}: must be a bool")
    return v


def _optional_nonnegative_int(
    d: dict[str, Any], key: str, location: str
) -> int | None:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ContractError(f"{location}.{key}: must be a non-negative integer or null")
    return v

