"""frame_jury.checks.vlm_scene — scene-level defect checking via local VLM.

SPEC.md §6: Runs under budget="full" when the VLM backend is available.
Evaluates scene-level defects that require semantic understanding of the shot declaration:
1. duplicated_character: whether any declared character appears more than once as copies
   of the same person.
2. missing_entity: whether any declared character or object is missing from the image.

Technique: VQAScore (one forward pass, first answer token logit ratio).
Verbalised JSON generation is deliberately avoided due to chance-level AUC (0.50).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from frame_jury.backends.base import VlmScorerBackend
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.contract import (
    DEFECT_DUPLICATED_CHARACTER,
    DEFECT_MISSING_ENTITY,
    Abstention,
    Entity,
    Finding,
    Shot,
)

_CHECK_NAME = "vlm_scene"


def format_declared_entities(
    shot: Shot, entities: tuple[Entity, ...] | list[Entity] | None = None
) -> str:
    """Format declared entities from the shot declaration.

    Each entity is formatted as:
        - <display_name> (<kind>): <visual_identity>
    truncated to 220 characters per entity line.
    Returns '- (no entities declared)' when empty.
    """
    entity_list = entities if entities is not None else shot.declared_entities
    if not entity_list:
        return "- (no entities declared)"
    lines = []
    for entity in entity_list:
        vis = entity.visual_identity.replace("\r\n", " ").replace("\n", " ")
        lines.append(f"- {entity.display_name} ({entity.kind}): {vis[:220]}")
    return "\n".join(lines)


def build_duplicated_character_question(shot: Shot) -> str:
    """Build the yes/no question for duplicated_character.

    Uses the exact generic wording measured in the lab (AUC 0.94).
    Collective characters are excluded from the declaration block.
    """
    entities_to_show = [
        e for e in shot.declared_entities
        if not (e.kind == "character" and e.is_collective)
    ]
    decl = format_declared_entities(shot, entities=entities_to_show)
    return (
        f"This is a frame from an AI-generated film. The shot declared these entities:\n{decl}\n\n"
        "Does any ONE of the declared characters appear more than once in the image, "
        "as two copies of the same person? Background extras and different people who merely "
        "dress alike do not count."
    )


def build_missing_entity_question(shot: Shot) -> str:
    """Build the yes/no question for missing_entity."""
    decl = format_declared_entities(shot)
    return (
        f"This is a frame from an AI-generated film. The shot declared these entities:\n{decl}\n\n"
        "Is any of the declared characters or objects missing from the image?"
    )


def run_vlm_scene_check(
    image_path: str | Path,
    shot: Shot,
    vlm_scorer: VlmScorerBackend | None,
    *,
    calibration: CalibrationFile | None = None,
) -> tuple[list[Finding], list[Abstention], dict[str, Any], float]:
    """Run the VLM scene check and return findings, abstentions, measurements, elapsed_ms.

    Parameters
    ----------
    image_path:
        The frame image to judge.
    shot:
        The shot declaration containing camera framing and declared entities.
    vlm_scorer:
        A VlmScorerBackend instance, or None if the backend is unavailable.
    calibration:
        CalibrationFile providing per-framing thresholds. If None, defaults are loaded.

    Returns
    -------
    findings:
        Findings produced if score > threshold.
    abstentions:
        Abstention emitted if backend is unavailable.
    measurements:
        Raw scores and thresholds.
    elapsed_ms:
        Elapsed wall time in milliseconds.
    """
    if calibration is None:
        calibration = load_defaults()

    t0 = time.perf_counter()
    thresholds = calibration.for_framing(shot.framing)

    if vlm_scorer is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        abstentions = [
            Abstention(
                check=_CHECK_NAME,
                reason="vlm_backend_unavailable",
                explanation="VLM scorer backend is unavailable under budget='full'",
            )
        ]
        return [], abstentions, {"vlm_backend_available": False}, elapsed_ms

    findings: list[Finding] = []
    abstentions: list[Abstention] = []
    measurements: dict[str, Any] = {"vlm_backend_available": True}

    model_name = vlm_scorer.name()
    model_rev = vlm_scorer.revision()

    # 1. duplicated_character
    characters = shot.characters()
    non_collective_chars = shot.non_collective_characters()

    if characters and not non_collective_chars:
        # Every declared character is collective: do not ask clone question; abstain.
        abstentions.append(
            Abstention(
                check=_CHECK_NAME,
                reason="all_characters_collective",
                explanation=(
                    "all declared characters are collective; "
                    "duplicated_character check does not apply to groups"
                ),
            )
        )
        measurements["vlm_duplicated_character_skipped"] = "all_characters_collective"
    else:
        q_dup = build_duplicated_character_question(shot)
        p_dup = vlm_scorer.score_yes_no(image_path, q_dup)
        thresh_dup = thresholds.vlm_duplicated_character_threshold
        measurements["vlm_duplicated_character_p_yes"] = p_dup
        measurements["vlm_duplicated_character_threshold"] = thresh_dup

        if p_dup > thresh_dup:
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_DUPLICATED_CHARACTER,
                    severity="warning",
                    confidence=p_dup,
                    evidence={
                        "question": q_dup,
                        "p_yes": p_dup,
                        "model": model_name,
                        "revision": model_rev,
                    },
                    explanation=(
                        f"VLM detected duplicated character with probability {p_dup:.4f} "
                        f"(threshold {thresh_dup:.4f})"
                    ),
                    prompt_hint="ensure each declared character appears at most once in the scene",
                )
            )

    # 2. missing_entity
    q_miss = build_missing_entity_question(shot)
    p_miss = vlm_scorer.score_yes_no(image_path, q_miss)
    thresh_miss = thresholds.vlm_missing_entity_threshold
    measurements["vlm_missing_entity_p_yes"] = p_miss
    measurements["vlm_missing_entity_threshold"] = thresh_miss

    if p_miss > thresh_miss:
        findings.append(
            Finding(
                check=_CHECK_NAME,
                defect=DEFECT_MISSING_ENTITY,
                severity="warning",
                confidence=p_miss,
                evidence={
                    "question": q_miss,
                    "p_yes": p_miss,
                    "model": model_name,
                    "revision": model_rev,
                },
                explanation=(
                    f"VLM detected missing entity with probability {p_miss:.4f} "
                    f"(threshold {thresh_miss:.4f})"
                ),
                prompt_hint="ensure all declared characters and objects are visible in the scene",
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return findings, abstentions, measurements, elapsed_ms
