"""frame_jury.backends.base — the interface every detector backend must honour.

SPEC.md §6: "Backends are replaceable. A check states what it needs; a backend
provides it. Swapping RT-DETR for something better must touch one file."

The presence check only needs :meth:`DetectorBackend.detect_people`.  Any
backend that satisfies this protocol can be dropped in without modifying the
check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One detected object instance.

    Attributes
    ----------
    label:
        The class label as a string (e.g. ``"person"``).
    confidence:
        Detection confidence score in [0.0, 1.0].
    box:
        Bounding box as ``(x_min, y_min, x_max, y_max)`` in pixel coordinates,
        with the origin at the top-left corner of the image.
    """

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max

    def to_list(self) -> list[int]:
        return list(self.box)


class DetectorBackend(ABC):
    """Abstract base for all object-detection backends.

    Concrete implementations live in this package: one file each, one class
    each.  The class must be constructible without arguments (all configuration
    via the defaulted calibration thresholds), and :meth:`detect` must be
    pure — same image, same result, every time.

    SPEC.md requirements:
    - No network at inference: weights are cached locally, pinned by sha256.
    - CPU path always exists: backends that default to GPU must fall back.
    - Deterministic: same image and declaration give the same verdict.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name + variant, e.g. ``'fasterrcnn-mobilenet-v3'``."""

    @abstractmethod
    def weights_sha256(self) -> str:
        """SHA-256 hex digest of the downloaded weights file.

        Used in the verdict's ``detectors`` list for traceability (SPEC.md §4).
        The sha256 is computed once on first download and cached; it is compared
        against the pinned value before inference.
        """

    @abstractmethod
    def detect(self, image_path: str | Path, *, score_threshold: float = 0.5) -> list[Detection]:
        """Detect all objects in *image_path* above *score_threshold*.

        Parameters
        ----------
        image_path:
            Absolute path to a JPEG or PNG image.  The backend opens it
            read-only; it never modifies the file.
        score_threshold:
            Drop detections with confidence below this value.  The caller
            (the presence check) passes the per-framing threshold from
            ``calibration/``.

        Returns
        -------
        list[Detection]
            All detected instances, sorted by confidence descending.
            Returns an empty list rather than raising when no objects are found.
        """

    # ── Convenience helper used by the presence check ──────────────────────

    def detect_people(
        self,
        image_path: str | Path,
        *,
        score_threshold: float = 0.5,
    ) -> list[Detection]:
        """Return only detections labelled ``"person"``.

        The presence check calls this, not :meth:`detect` directly, so that
        the person-class label difference between backends is invisible to the
        check logic.
        """
        return [
            d
            for d in self.detect(image_path, score_threshold=score_threshold)
            if d.label == "person"
        ]


class FaceBackend(ABC):
    """Abstract base for all face detection and embedding backends.

    Concrete implementations live in this package: one file each, one class
    each. The class must be constructible without arguments (all configuration
    via defaulted calibration thresholds), and methods must be pure — same
    image, same result, every time.

    SPEC.md requirements:
    - No network at inference: weights are cached locally, pinned by sha256.
    - CPU path always exists: runs on CPU.
    - Deterministic: same image gives same detections and embeddings.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name + variant, e.g. ``'yunet-sface'``."""

    @abstractmethod
    def weights_sha256(self) -> str:
        """SHA-256 hex digest of the downloaded weights file(s).

        Used in the verdict's ``detectors`` list for traceability (SPEC.md §4).
        """

    @abstractmethod
    def detect_faces(self, image_path: str | Path) -> list[Detection]:
        """Detect all faces in *image_path*.

        Parameters
        ----------
        image_path:
            Absolute path to a JPEG or PNG image. The backend opens it
            read-only; it never modifies the file.

        Returns
        -------
        list[Detection]
            All detected face instances, sorted by confidence descending.
            Each Detection has label ``"face"``.
            Returns an empty list rather than raising when no faces are found.
        """

    @abstractmethod
    def embed(
        self,
        image_path: str | Path,
        box: tuple[int, int, int, int],
    ) -> list[float]:
        """Compute the facial feature embedding for the face inside *box*.

        Parameters
        ----------
        image_path:
            Absolute path to the image containing the face.
        box:
            Bounding box as ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.

        Returns
        -------
        list[float]
            The feature embedding vector.
        """

    @abstractmethod
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two face embeddings.

        Parameters
        ----------
        a:
            First embedding vector.
        b:
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity in [-1.0, 1.0], where 1.0 means identical,
            0.0 means orthogonal, -1.0 means opposite, and higher means
            more alike.
        """
