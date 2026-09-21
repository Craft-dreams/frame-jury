"""frame_jury.checks.presence — compare detections to the shot declaration.

This check is the core idea of the project (AGENTS.md §the-one-idea):

    A detector counts people.  The *declaration* decides whether that count
    is a defect.  A frame with two people is correct when the shot declared
    two and a defect when it declared one.

The check never modifies the image or the declaration.  It reads the
detector's output and compares it to:
  - ``shot.declared_entities`` (which entities must be visible, of what kind);
  - ``shot.framing`` (a close-up and a wide shot fail differently).

Defects produced (SPEC.md §5):
  ``duplicated_character``
      More people detected than characters declared.  The shot said one person;
      the frame has two — contract violation.

  ``extra_person``
      At least one person detected, but the shot declared *no* characters at
      all.  The declaration is "no person"; any person in the frame is extra.

  ``missing_entity``
      A character or named object is declared but the detector finds nobody.
      Only characters trigger the person-count path; objects are out of scope
      for the person detector (they would need a dedicated detector) and are
      noted as ``unsure`` when no object-specific backend is present.

``unsure`` semantics (SPEC.md §4):
    "A judge that guesses is worse than one that abstains."  When the detector
    finds zero people in a frame that declares a character, the default
    calibration emits ``unsure``, not ``missing_entity``.  A blank image and a
    wide-angle extreme shot both return zero detections; until labels teach us
    which is which, we abstain.  The calibration key
    ``missing_entity_abstain_on_empty`` controls this.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from frame_jury.backends.base import Detection, DetectorBackend
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.contract import (
    DEFECT_DUPLICATED_CHARACTER,
    DEFECT_EXTRA_PERSON,
    DEFECT_MISSING_ENTITY,
    Abstention,
    Finding,
    Shot,
)

_CHECK_NAME = "presence"


def run_presence_check(
    image_path: str | Path,
    shot: Shot,
    detector: DetectorBackend,
    *,
    calibration: CalibrationFile | None = None,
) -> tuple[list[Finding], list[Abstention], dict[str, Any], float]:
    """Run the presence check and return findings, abstentions, measurements, elapsed_ms.

    Parameters
    ----------
    image_path:
        The frame to judge.
    shot:
        The shot declaration — which entities must be visible, of what kind,
        and the camera framing.
    detector:
        A :class:`~frame_jury.backends.base.DetectorBackend` that provides
        person detection.  Must already be loaded; this function does not
        construct or download anything.
    calibration:
        A :class:`~frame_jury.calibration.thresholds.CalibrationFile`.
        If *None*, the shipped defaults are used.

    Returns
    -------
    findings:
        Zero or more :class:`~frame_jury.contract.Finding` instances.
    abstentions:
        Zero or more :class:`~frame_jury.contract.Abstention` instances.
    measurements:
        A dict of raw numbers to merge into the verdict's ``measurements``
        field (e.g. ``{"people": 2, "declared_characters": 1}``).
    elapsed_ms:
        Wall time for this check in milliseconds.
    """
    if calibration is None:
        calibration = load_defaults()

    t0 = time.perf_counter()

    thresholds = calibration.for_framing(shot.framing)
    declared_characters = shot.declared_character_count
    characters = shot.characters()

    # ── Run the detector ────────────────────────────────────────────────────
    detections: list[Detection] = detector.detect_people(
        image_path,
        score_threshold=thresholds.person_score_threshold,
    )
    people_detected = len(detections)

    # ── Build evidence dict (always present in the finding, SPEC.md §4) ────
    boxes = [d.to_list() for d in detections]

    measurements: dict[str, Any] = {
        "people_detected": people_detected,
        "declared_characters": declared_characters,
        "person_score_threshold": thresholds.person_score_threshold,
    }

    findings: list[Finding] = []
    abstentions: list[Abstention] = []

    # ── Case 1: no character declared ───────────────────────────────────────
    # If the shot declares no characters at all but the detector finds a person,
    # that is an extra_person defect.
    if declared_characters == 0:
        if people_detected > 0:
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_EXTRA_PERSON,
                    severity="blocking",
                    confidence=thresholds.extra_person_confidence,
                    evidence={
                        "people_detected": people_detected,
                        "declared_characters": 0,
                        "boxes": boxes,
                    },
                    explanation=(
                        f"the shot declares no characters but "
                        f"{people_detected} person(s) detected in frame"
                    ),
                    prompt_hint=(
                        "add a person to the negative prompt, or declare "
                        "the character in the shot"
                    ),
                )
            )
        # No characters declared and none detected: clean for this check.
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return findings, abstentions, measurements, elapsed_ms

    # ── Case 2: characters declared ─────────────────────────────────────────

    # 2a. More people than declared → duplicated_character
    if people_detected > declared_characters:
        extra = people_detected - declared_characters
        # Report against each declared character entity (the declaration is the
        # contract; we don't know which detection is the duplicate).
        entity_id = characters[0].entity_id if characters else ""
        findings.append(
            Finding(
                check=_CHECK_NAME,
                defect=DEFECT_DUPLICATED_CHARACTER,
                severity="blocking",
                confidence=thresholds.duplicated_character_confidence,
                evidence={
                    "people_detected": people_detected,
                    "declared_characters": declared_characters,
                    "extra_count": extra,
                    "boxes": boxes,
                },
                explanation=(
                    f"the shot declares {declared_characters} character(s) but "
                    f"{people_detected} people detected in frame"
                ),
                entity_id=entity_id,
                prompt_hint=(
                    f"state that exactly {declared_characters} person(s) "
                    "are in frame; add the extra person(s) to the negative prompt"
                ),
            )
        )

    # 2b. Zero people detected but characters declared → missing_entity or unsure
    elif people_detected == 0:
        if thresholds.missing_entity_abstain_on_empty:
            # Abstain: we cannot tell a blank image from an extreme wide shot
            # without labels. Emit unsure via a first-class Abstention (SPEC.md §4).
            measurements["presence_abstain"] = True
            measurements["presence_abstain_reason"] = "no_person_in_frame"
            measurements["abstain_reason"] = "no_person_in_frame"
            for entity in characters:
                abstentions.append(
                    Abstention(
                        check=_CHECK_NAME,
                        reason="no_person_in_frame",
                        entity_id=entity.entity_id,
                        explanation=(
                            f"no person detected in frame; "
                            f"'{entity.display_name}' was declared — "
                            "detector abstains on zero detections (calibration: "
                            "missing_entity_abstain_on_empty=true)"
                        ),
                    )
                )
        else:
            # Reject: calibration says zero detections with declared characters
            # is always a defect.
            for entity in characters:
                findings.append(
                    Finding(
                        check=_CHECK_NAME,
                        defect=DEFECT_MISSING_ENTITY,
                        severity="blocking",
                        confidence=thresholds.missing_entity_confidence,
                        evidence={
                            "people_detected": 0,
                            "declared_characters": declared_characters,
                            "boxes": [],
                        },
                        explanation=(
                            f"no person detected in frame; "
                            f"'{entity.display_name}' was declared"
                        ),
                        entity_id=entity.entity_id,
                        prompt_hint=(
                            "ensure the character is visible and not obscured"
                        ),
                    )
                )

    # 2c. Correct count (people_detected == declared_characters): no finding.
    # Note: we do not check which people are the right ones here; that is M3
    # (identity check).

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return findings, abstentions, measurements, elapsed_ms
