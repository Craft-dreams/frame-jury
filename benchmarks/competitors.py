"""benchmarks.competitors — competitor interfaces and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from frame_jury.backends.base import DetectorBackend, FaceBackend
from frame_jury.calibration.thresholds import CalibrationFile
from frame_jury.contract import Abstention, Finding, JuryRequest, Shot, Verdict
from frame_jury.jury import judge


@dataclass(frozen=True)
class CompetitorResult:
    defects: frozenset[str]      # defect slugs this system claims
    abstained: frozenset[str]    # slugs it explicitly declined to judge


class Competitor(Protocol):
    name: str
    licence: str                 # SPDX id, e.g. "Apache-2.0"

    def judge_case(self, case: dict[str, Any]) -> CompetitorResult:
        ...


class NullCompetitor:
    """The floor everything must beat: always accepts, claims no defects, never abstains."""

    name: str = "null"
    licence: str = "n/a"

    def judge_case(self, case: dict[str, Any]) -> CompetitorResult:
        return CompetitorResult(defects=frozenset(), abstained=frozenset())


# Map check name to defect slug for Abstention.
# Abstention has no defect field; must be extended when a new check starts abstaining.
_ABSTENTION_DEFECT: dict[str, str] = {
    "identity": "wrong_identity",
    "presence": "missing_entity",
}


def _defect_for_abstention(abstention: Abstention) -> str | None:
    """Map an Abstention to its defect slug.

    PR #6 introduced first-class Abstention(check, reason, entity_id, explanation).
    Unknown checks return None so they never enter CompetitorResult.abstained as
    fake defect slugs.
    """
    return _ABSTENTION_DEFECT.get(abstention.check)


class FrameJuryCompetitor:
    """Our system running at budget="cheap"."""

    name: str = "frame-jury"
    licence: str = "AGPL-3.0-or-later"

    def __init__(
        self,
        *,
        detector: DetectorBackend | None = None,
        face_backend: FaceBackend | None = None,
        calibration: CalibrationFile | None = None,
    ) -> None:
        self.detector = detector
        self.face_backend = face_backend
        self.calibration = calibration

    def judge_case(self, case: dict[str, Any]) -> CompetitorResult:
        if self.detector is None:
            from frame_jury.jury import _resolve_detector

            self.detector = _resolve_detector(None)
        if self.face_backend is None:
            from frame_jury.jury import _resolve_face_backend

            self.face_backend = _resolve_face_backend(None)

        request = JuryRequest(
            schema_version="2.0",
            image_path=case["image_path"],
            shot=Shot.from_dict(case["shot"]),
            checks=(),
            budget="cheap",
        )
        verdict: Verdict = judge(
            request,
            detector=self.detector,
            face_backend=self.face_backend,
            calibration=self.calibration,
        )

        defects = frozenset(f.defect for f in verdict.findings)
        abstained = frozenset(
            d for a in verdict.abstentions if (d := _defect_for_abstention(a)) is not None
        )
        return CompetitorResult(defects=defects, abstained=abstained)
