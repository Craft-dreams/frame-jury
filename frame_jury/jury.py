"""frame_jury.jury — the router: runs checks in order, stops early, merges findings.

SPEC.md §6 architecture:
  "The router runs deterministic checks first and only escalates what they
  cannot settle, which is also how the factory orders its own work."

Cheap gates first (SPEC.md §6, principle 1):
  presence  → cheap, ~100 ms on CPU
  identity  → M3, not yet implemented
  anatomy   → M5, not yet implemented
  legibility → M5, not yet implemented

The router:
1. Accepts a :class:`~frame_jury.contract.JuryRequest`.
2. Decides which checks to run (from ``request.checks``), always in the
   cheap-first order.
3. For each check, calls the check function with the appropriate backend.
4. Stops early if a ``blocking`` finding is produced and ``budget=cheap``.
   (Under ``budget=full``, all checks run and the VLM is called last.)
5. Merges findings into a :class:`~frame_jury.contract.Verdict` via
   :class:`~frame_jury.contract.VerdictBuilder`.

Backend selection
-----------------
The default backend is :class:`~frame_jury.backends.detector_torchvision.TorchvisionDetector`
(FasterRCNN-MobileNetV3, BSD-3-Clause).  Pass *detector* to override, e.g. in
tests or to use :class:`~frame_jury.backends.detector_rtdetr.RTDetrDetector`.

Early stop
----------
``budget=cheap``: stop after the first check that produces a ``blocking``
finding.  This mirrors the factory's own rule that expensive work follows
cheaper gates (SPEC.md §6).

``budget=full``: run all requested checks regardless.  The optional VLM
backend (M-future) would be called here; it is a no-op in M2.
"""

from __future__ import annotations

from pathlib import Path

from frame_jury.backends.base import DetectorBackend, FaceBackend
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.checks.identity import run_identity_check
from frame_jury.checks.presence import run_presence_check
from frame_jury.contract import JuryRequest, Verdict, VerdictBuilder

# ── Check order (SPEC.md §6, cheap first) ──────────────────────────────────
_CHEAP_FIRST_ORDER = ["presence", "identity", "anatomy", "legibility"]


def judge(
    request: JuryRequest,
    *,
    detector: DetectorBackend | None = None,
    face_backend: FaceBackend | None = None,
    calibration: CalibrationFile | None = None,
) -> Verdict:
    """Judge one frame against its declaration.

    Parameters
    ----------
    request:
        Parsed request; see :func:`~frame_jury.contract.JuryRequest.from_json`.
    detector:
        Object-detection backend.  If *None*, the torchvision backend is
        constructed on first use (lazy import so tests that mock the detector
        never load torch).
    face_backend:
        Face detection and embedding backend. If *None*, the YuNet+SFace backend
        is constructed on first use (lazy import so tests that mock the backend
        never load OpenCV).
    calibration:
        Calibration thresholds.  If *None*, the shipped defaults are used.

    Returns
    -------
    Verdict
        The merged verdict from all checks that ran.
    """
    if calibration is None:
        calibration = load_defaults()

    builder = VerdictBuilder(shot_id=request.shot.shot_id)
    checks_requested = set(request.checks) if request.checks else set(_CHEAP_FIRST_ORDER)
    budget = request.budget

    # Resolve backends lazily so tests can avoid importing heavy dependencies.
    resolved_detector: DetectorBackend | None = None
    resolved_face_backend: FaceBackend | None = None

    # Run checks in cheap-first order.
    for check_name in _CHEAP_FIRST_ORDER:
        if check_name not in checks_requested:
            continue

        if check_name == "presence":
            if resolved_detector is None:
                resolved_detector = _resolve_detector(detector)
            findings, measurements, elapsed_ms = run_presence_check(
                request.image_path,
                request.shot,
                resolved_detector,
                calibration=calibration,
            )
            builder.add_findings(findings)
            builder.update_measurements(measurements)
            builder.record_timing("presence", elapsed_ms)
            builder.record_detector(
                "presence",
                resolved_detector.name(),
                resolved_detector.weights_sha256(),
            )

        elif check_name == "identity":
            if resolved_face_backend is None:
                resolved_face_backend = _resolve_face_backend(face_backend)
            findings, measurements, elapsed_ms = run_identity_check(
                request.image_path,
                request.shot,
                resolved_face_backend,
                calibration=calibration,
            )
            builder.add_findings(findings)
            builder.update_measurements(measurements)
            builder.record_timing("identity", elapsed_ms)
            builder.record_detector(
                "identity",
                resolved_face_backend.name(),
                resolved_face_backend.weights_sha256(),
            )

        elif check_name == "anatomy":
            # M5 — not yet implemented.
            pass

        elif check_name == "legibility":
            # M5 — not yet implemented.
            pass

        # Early stop on blocking findings when budget is cheap.
        if budget == "cheap" and any(
            f.severity == "blocking" for f in builder._findings  # noqa: SLF001
        ):
            break

    return builder.build()


# ── Private helpers ─────────────────────────────────────────────────────────


def _resolve_detector(detector: DetectorBackend | None) -> DetectorBackend:
    """Return *detector* if provided, otherwise construct the default backend.

    The default is torchvision (BSD-3-Clause).  The import is deferred so that
    tests that pass a stub detector never import torch.
    """
    if detector is not None:
        return detector
    from frame_jury.backends.detector_torchvision import TorchvisionDetector

    return TorchvisionDetector()


def _resolve_face_backend(face_backend: FaceBackend | None) -> FaceBackend:
    """Return *face_backend* if provided, otherwise construct the default backend.

    The default is YuNet+SFace (MIT / Apache-2.0). The import is deferred so that
    tests that pass a stub face backend never import cv2.
    """
    if face_backend is not None:
        return face_backend
    from frame_jury.backends.face_yunet_sface import YuNetSFaceBackend

    return YuNetSFaceBackend()
