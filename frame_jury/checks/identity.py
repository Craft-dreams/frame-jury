"""frame_jury.checks.identity — reference-vs-frame face identity check.

This check embodies the core idea of frame-jury (SPEC.md §1, AGENTS.md §the-one-idea):
We are not building a blind face recogniser. We have an approved reference image
(Entity.reference_images) of exactly who the character is supposed to be.
The reference is the anchor; the model only measures distance to it.

Defect produced (SPEC.md §5):
  ``wrong_identity``:
      Face embedding similarity between the frame and the declared character's
      approved reference image is below the threshold for the shot's camera framing.

Abstention semantics (SPEC.md §4):
  "A judge that guesses is worse than one that abstains."
  The check abstains (emitting unsure via a warning-severity finding) when:
  - no face is found in the frame;
  - no face is found in the reference image;
  - the entity declares no reference image;
  - or the similarity sits inside an ambiguous band around the threshold.
  Each abstention states its reason in measurements.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from frame_jury.backends.base import FaceBackend
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.contract import (
    DEFECT_WRONG_IDENTITY,
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
) -> tuple[list[Finding], dict[str, Any], float]:
    """Run the identity check and return findings, measurements, elapsed_ms.

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
        return [], measurements, elapsed_ms

    findings: list[Finding] = []

    # Detect faces in the frame.
    frame_faces = face_backend.detect_faces(image_path)
    measurements["faces_detected"] = len(frame_faces)
    measurements["faces"] = len(frame_faces)

    # Frame face boxes:
    frame_boxes = [f.to_list() for f in frame_faces]

    for entity in characters:
        entity_id = entity.entity_id

        # Abstain Case 1: Entity declares no reference image.
        if not entity.reference_images:
            measurements["identity_abstain"] = True
            measurements["identity_abstain_reason"] = "no_reference_image"
            measurements["abstain_reason"] = "no_reference_image"
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_WRONG_IDENTITY,
                    severity="warning",  # → unsure
                    confidence=confidence,
                    evidence={
                        "entity_id": entity_id,
                        "reference_images": [],
                    },
                    explanation=(
                        f"entity '{entity.display_name}' declares no reference "
                        f"image; identity check abstains"
                    ),
                    entity_id=entity_id,
                    prompt_hint=(
                        f"provide an approved reference image for '{entity.display_name}' "
                        f"in the production bible"
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
            measurements["identity_abstain"] = True
            measurements["identity_abstain_reason"] = "no_face_in_reference"
            measurements["abstain_reason"] = "no_face_in_reference"
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_WRONG_IDENTITY,
                    severity="warning",  # → unsure
                    confidence=confidence,
                    evidence={
                        "entity_id": entity_id,
                        "reference_images": list(entity.reference_images),
                        "faces_in_reference": 0,
                    },
                    explanation=(
                        f"no face detected in reference image(s) for '{entity.display_name}'; "
                        f"identity check abstains"
                    ),
                    entity_id=entity_id,
                    prompt_hint=(
                        f"ensure the reference image for '{entity.display_name}' "
                        f"clearly shows the character's face"
                    ),
                )
            )
            continue

        # Abstain Case 3: No face found in frame.
        if not frame_faces:
            measurements["identity_abstain"] = True
            measurements["identity_abstain_reason"] = "no_face_in_frame"
            measurements["abstain_reason"] = "no_face_in_frame"
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_WRONG_IDENTITY,
                    severity="warning",  # → unsure
                    confidence=confidence,
                    evidence={
                        "entity_id": entity_id,
                        "faces_detected": 0,
                        "boxes": [],
                    },
                    explanation=(
                        f"no face detected in frame; cannot verify identity for "
                        f"'{entity.display_name}'"
                    ),
                    entity_id=entity_id,
                    prompt_hint=(
                        f"ensure the character's face is visible in frame, "
                        f"or check whether framing obscured the subject"
                    ),
                )
            )
            continue

        # Both sides have faces: embed and compute pairwise similarity.
        best_similarity = -2.0
        best_ref_path = ""
        best_ref_box: tuple[int, int, int, int] = (0, 0, 0, 0)
        best_frame_box: tuple[int, int, int, int] = (0, 0, 0, 0)

        # Cache reference embeddings for candidate faces.
        ref_embeddings = [
            (r_path, rf.box, face_backend.embed(r_path, rf.box))
            for r_path, rf in ref_face_candidates
        ]

        # Embed frame faces and find maximum similarity.
        for ff in frame_faces:
            frame_emb = face_backend.embed(image_path, ff.box)
            for r_path, r_box, r_emb in ref_embeddings:
                sim = face_backend.similarity(frame_emb, r_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_ref_path = r_path
                    best_ref_box = r_box
                    best_frame_box = ff.box

        measurements["identity_similarity"] = round(best_similarity, 4)

        lower_band = threshold - band
        upper_band = threshold + band

        # Abstain Case 4: Similarity sits in the ambiguous band around threshold.
        if lower_band <= best_similarity <= upper_band:
            measurements["identity_abstain"] = True
            measurements["identity_abstain_reason"] = "ambiguous_band"
            measurements["abstain_reason"] = "ambiguous_band"
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_WRONG_IDENTITY,
                    severity="warning",  # → unsure
                    confidence=confidence,
                    evidence={
                        "similarity": round(best_similarity, 4),
                        "threshold": threshold,
                        "ambiguous_band": band,
                        "framing": shot.framing,
                        "reference_path": best_ref_path,
                        "reference_image": best_ref_path,
                        "boxes": [list(best_frame_box)],
                        "all_frame_boxes": frame_boxes,
                        "reference_box": list(best_ref_box),
                    },
                    explanation=(
                        f"face similarity {best_similarity:.4f} is inside ambiguous band "
                        f"[{lower_band:.4f}, {upper_band:.4f}] around threshold {threshold:.4f} "
                        f"for {shot.framing}; identity check abstains"
                    ),
                    entity_id=entity_id,
                    prompt_hint=(
                        f"similarity for '{entity.display_name}' is borderline; inspect frame "
                        f"or provide additional reference views"
                    ),
                )
            )

        # Case 5: Below threshold → wrong_identity (blocking).
        elif best_similarity < lower_band:
            findings.append(
                Finding(
                    check=_CHECK_NAME,
                    defect=DEFECT_WRONG_IDENTITY,
                    severity="blocking",  # → reject
                    confidence=confidence,
                    evidence={
                        "similarity": round(best_similarity, 4),
                        "threshold": threshold,
                        "framing": shot.framing,
                        "reference_path": best_ref_path,
                        "reference_image": best_ref_path,
                        "boxes": [list(best_frame_box)],
                        "all_frame_boxes": frame_boxes,
                        "reference_box": list(best_ref_box),
                    },
                    explanation=(
                        f"character '{entity.display_name}' face similarity {best_similarity:.4f} "
                        f"is below threshold {threshold:.4f} for {shot.framing} framing"
                    ),
                    entity_id=entity_id,
                    prompt_hint=(
                        f"strengthen visual identity cues for '{entity.display_name}' in prompt, "
                        f"or check for conflicting styling in negative prompt"
                    ),
                )
            )

        # Case 6: Above upper_band → clean match, no defect finding.

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return findings, measurements, elapsed_ms
