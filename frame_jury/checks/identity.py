"""frame_jury.checks.identity — reference-vs-frame face identity check.

This check embodies the core idea of frame-jury (SPEC.md §1, AGENTS.md §the-one-idea):
We are not building a blind face recogniser. We have an approved reference image
(Entity.reference_images) of exactly who the character is supposed to be.
The reference is the anchor; the model only measures distance to it.

Assignment semantics (SPEC.md §4, Decision 1):
  Identity assignment is bipartite and optimal, not greedy. We compute the
  similarity matrix between declared characters and detected faces, and solve
  for the optimal one-to-one assignment that maximizes total similarity using
  the Hungarian algorithm (Kuhn-Munkres). Each declared character is judged
  against its assigned face only.
  A character left unassigned because there were fewer faces than characters
  is an abstention, not a wrong_identity defect.

Abstention semantics (SPEC.md §4, Decision 2):
  "A judge that guesses is worse than one that abstains."
  An abstention is a distinct, first-class field rather than a warning-severity finding.
  The check abstains (emitting unsure via an Abstention) when:
  - no face is found in the frame (reason: "no_face_in_frame");
  - no face is found in the reference image (reason: "no_face_in_reference");
  - the entity declares no reference image (reason: "no_reference_image");
  - the similarity sits inside an ambiguous band around the threshold (reason: "ambiguous_band");
  - or fewer faces than characters leave the surplus unassigned (reason: "unassigned_face").
  Each abstention states its reason in measurements.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from frame_jury.backends.base import FaceBackend
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.checks.assignment import best_assignment
from frame_jury.contract import (
    DEFECT_WRONG_IDENTITY,
    Abstention,
    Finding,
    Shot,
)

_CHECK_NAME = "identity"


def run_identity_check(
    image_path: str | Path,
    shot: Shot,
    face_backend: FaceBackend,
    *,
    calibration: CalibrationFile | None = None,
) -> tuple[list[Finding], list[Abstention], dict[str, Any], float]:
    """Run the identity check and return findings, abstentions, measurements, elapsed_ms.

    Parameters
    ----------
    image_path:
        The frame to judge.
    shot:
        The shot declaration — declared entities, reference images, framing.
    face_backend:
        A :class:`~frame_jury.backends.base.FaceBackend` providing face detection
        and embedding extraction.
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
        Raw numbers and abstention reasons for the verdict ledger.
    elapsed_ms:
        Wall time for this check in milliseconds.
    """
    if calibration is None:
        calibration = load_defaults()

    t0 = time.perf_counter()

    thresholds = calibration.for_framing(shot.framing)
    threshold = thresholds.identity_similarity_threshold
    band = thresholds.identity_ambiguous_band
    confidence = thresholds.identity_confidence

    characters = shot.characters()
    measurements: dict[str, Any] = {
        "declared_characters": len(characters),
        "identity_threshold": threshold,
        "identity_framing": shot.framing,
        "identity_ambiguous_band": band,
    }

    # Case 0: No characters declared in this shot.
    if not characters:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return [], [], measurements, elapsed_ms

    findings: list[Finding] = []
    abstentions: list[Abstention] = []

    # Detect faces in the frame.
    frame_faces = face_backend.detect_faces(image_path)
    measurements["faces_detected"] = len(frame_faces)
    measurements["faces"] = len(frame_faces)

    frame_boxes = [f.to_list() for f in frame_faces]

    # Partition declared characters: those with valid reference faces vs those that abstain upfront.
    candidate_characters: list[tuple[Any, list[tuple[str, Any]]]] = []

    for entity in characters:
        entity_id = entity.entity_id

        # Abstain Case 1: Entity declares no reference image.
        if not entity.reference_images:
            abstentions.append(
                Abstention(
                    check=_CHECK_NAME,
                    reason="no_reference_image",
                    entity_id=entity_id,
                    explanation=(
                        f"entity '{entity.display_name}' declares no reference "
                        f"image; identity check abstains"
                    ),
                )
            )
            continue

        # Detect faces across all reference images for this character.
        ref_face_candidates: list[tuple[str, Any]] = []
        for ref_img in entity.reference_images:
            faces_in_ref = face_backend.detect_faces(ref_img)
            for rf in faces_in_ref:
                ref_face_candidates.append((str(ref_img), rf))

        # Abstain Case 2: No face found in reference image(s).
        if not ref_face_candidates:
            abstentions.append(
                Abstention(
                    check=_CHECK_NAME,
                    reason="no_face_in_reference",
                    entity_id=entity_id,
                    explanation=(
                        f"no face detected in reference image(s) for '{entity.display_name}'; "
                        f"identity check abstains"
                    ),
                )
            )
            continue

        candidate_characters.append((entity, ref_face_candidates))

    # Abstain Case 3: No face found in frame for candidate characters.
    if candidate_characters and not frame_faces:
        for entity, _ in candidate_characters:
            abstentions.append(
                Abstention(
                    check=_CHECK_NAME,
                    reason="no_face_in_frame",
                    entity_id=entity.entity_id,
                    explanation=(
                        f"no face detected in frame; cannot verify identity for "
                        f"'{entity.display_name}'"
                    ),
                )
            )

    # Both candidate characters and frame faces exist: perform optimal bipartite assignment.
    elif candidate_characters and frame_faces:
        # Pre-embed reference faces for each candidate character.
        ref_embeddings: list[list[tuple[str, tuple[int, int, int, int], list[float]]]] = []
        for _, ref_faces in candidate_characters:
            ref_embeddings.append([
                (r_path, rf.box, face_backend.embed(r_path, rf.box))
                for r_path, rf in ref_faces
            ])

        # Pre-embed all detected frame faces once.
        frame_embeddings = [
            face_backend.embed(image_path, ff.box)
            for ff in frame_faces
        ]

        # Build similarity matrix (cost[i][j] is similarity between character i and frame face j).
        cost_matrix: list[list[float]] = []
        pair_details: list[list[tuple[str, tuple[int, int, int, int]]]] = []

        for i in range(len(candidate_characters)):
            row_sims: list[float] = []
            row_details: list[tuple[str, tuple[int, int, int, int]]] = []
            for j in range(len(frame_faces)):
                best_sim = -2.0
                best_ref_path = ""
                best_ref_box = (0, 0, 0, 0)
                for r_path, r_box, r_emb in ref_embeddings[i]:
                    sim = face_backend.similarity(frame_embeddings[j], r_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_ref_path = r_path
                        best_ref_box = r_box
                row_sims.append(best_sim)
                row_details.append((best_ref_path, best_ref_box))
            cost_matrix.append(row_sims)
            pair_details.append(row_details)

        # Optimal one-to-one assignment via Hungarian algorithm (Decision 1).
        assignments = best_assignment(cost_matrix)
        char_to_face = {char_idx: face_idx for char_idx, face_idx in assignments}

        assigned_similarities: list[float] = []
        lower_band = threshold - band
        upper_band = threshold + band

        for i, (entity, _) in enumerate(candidate_characters):
            entity_id = entity.entity_id

            if i not in char_to_face:
                # Rectangular surplus: more characters than faces in frame.
                # Character left unassigned is an abstention, not a wrong_identity.
                abstentions.append(
                    Abstention(
                        check=_CHECK_NAME,
                        reason="unassigned_face",
                        entity_id=entity_id,
                        explanation=(
                            f"fewer faces than characters in frame; "
                            f"'{entity.display_name}' was left unassigned"
                        ),
                    )
                )
                continue

            j = char_to_face[i]
            sim = cost_matrix[i][j]
            assigned_similarities.append(sim)
            ref_path, ref_box = pair_details[i][j]
            frame_box = frame_faces[j].box

            # Abstain Case 4: Similarity sits in ambiguous band around threshold.
            if lower_band <= sim <= upper_band:
                abstentions.append(
                    Abstention(
                        check=_CHECK_NAME,
                        reason="ambiguous_band",
                        entity_id=entity_id,
                        explanation=(
                            f"face similarity {sim:.4f} is inside ambiguous band "
                            f"[{lower_band:.4f}, {upper_band:.4f}] around threshold {threshold:.4f} "
                            f"for {shot.framing}; identity check abstains"
                        ),
                    )
                )

            # Case 5: Below threshold → wrong_identity (blocking defect).
            elif sim < lower_band:
                findings.append(
                    Finding(
                        check=_CHECK_NAME,
                        defect=DEFECT_WRONG_IDENTITY,
                        severity="blocking",
                        confidence=confidence,
                        evidence={
                            "similarity": round(sim, 4),
                            "threshold": threshold,
                            "framing": shot.framing,
                            "reference_path": ref_path,
                            "reference_image": ref_path,
                            "boxes": [list(frame_box)],
                            "all_frame_boxes": frame_boxes,
                            "reference_box": list(ref_box),
                        },
                        explanation=(
                            f"character '{entity.display_name}' face similarity {sim:.4f} "
                            f"is below threshold {threshold:.4f} for {shot.framing} framing"
                        ),
                        entity_id=entity_id,
                        prompt_hint=(
                            f"strengthen visual identity cues for '{entity.display_name}' in prompt, "
                            f"or check for conflicting styling in negative prompt"
                        ),
                    )
                )

            # Case 6: Above upper_band → clean match, no defect and no abstention.

        if assigned_similarities:
            measurements["identity_similarity"] = round(min(assigned_similarities), 4)

    if abstentions:
        measurements["identity_abstain"] = True
        measurements["identity_abstain_reason"] = abstentions[0].reason
        measurements["abstain_reason"] = abstentions[0].reason

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return findings, abstentions, measurements, elapsed_ms
