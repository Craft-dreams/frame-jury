"""frame_jury.calibration.thresholds — load and look up per-framing thresholds.

Thresholds are data, not code (SPEC.md §6).  They are stored in
``defaults.json`` (or a user-supplied override file) and loaded once at
startup.  The fits that will replace the defaults are produced by the harness
(M4) once labels exist (M7).

The loader is deliberately simple: one JSON file, keyed by framing string,
with ``_default`` as the fallback.  The fitting script writes a new JSON file
in the same format; nothing in this module changes.

Calibration status (M2, 2026-09-21)
-------------------------------------
All thresholds in ``defaults.json`` are **unfitted documentation values**.
No precision, recall or quality number may be claimed against them
(AGENTS.md §judging-honestly).  The ``fitted`` flag in the JSON is ``false``
and will become ``true`` only when the dev-split fitting run completes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The shipped defaults file, always present in the installed package.
_DEFAULTS_PATH = Path(__file__).parent / "defaults.json"


@dataclass(frozen=True)
class FramingThresholds:
    """All thresholds for one camera framing."""

    person_score_threshold: float
    missing_entity_abstain_on_empty: bool
    extra_person_confidence: float
    duplicated_character_confidence: float  # reserved for identity-based duplicate detection (SPEC §5)
    missing_entity_confidence: float

    # Identity thresholds (M3)
    identity_similarity_threshold: float = 0.55
    identity_ambiguous_band: float = 0.05
    identity_confidence: float = 0.85

    # VLM scene thresholds (I1)
    vlm_duplicated_character_threshold: float = 0.10  # calibrated: AUC 0.940, 4 positives (0.0015, 0.148, 0.182, 0.321), 3/4 recalled at 0.10 (precision 0.27, 3.8% FPR); not enough labels to calibrate a blocking gate
    vlm_missing_entity_threshold: float = 0.50  # calibrated: AUC 0.779, 3 positives (0.011, 0.269, 0.679); not enough labels to calibrate a blocking gate

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FramingThresholds":
        return cls(
            person_score_threshold=float(d["person_score_threshold"]),
            missing_entity_abstain_on_empty=bool(d["missing_entity_abstain_on_empty"]),
            extra_person_confidence=float(d["extra_person_confidence"]),
            duplicated_character_confidence=float(d["duplicated_character_confidence"]),
            missing_entity_confidence=float(d["missing_entity_confidence"]),
            identity_similarity_threshold=float(d.get("identity_similarity_threshold", 0.55)),
            identity_ambiguous_band=float(d.get("identity_ambiguous_band", 0.05)),
            identity_confidence=float(d.get("identity_confidence", 0.85)),
            vlm_duplicated_character_threshold=float(
                d.get("vlm_duplicated_character_threshold", 0.10)
            ),
            vlm_missing_entity_threshold=float(
                d.get("vlm_missing_entity_threshold", 0.50)
            ),
        )


@dataclass(frozen=True)
class CalibrationFile:
    """Parsed calibration file."""

    fitted: bool
    fitted_at: str | None       # ISO-8601 timestamp, or None if unfitted
    label_file: str | None      # path to the label file used for fitting
    thresholds: dict[str, FramingThresholds]  # keyed by framing, '_default' always present

    def for_framing(self, framing: str) -> FramingThresholds:
        """Return thresholds for *framing*, falling back to ``_default``."""
        return self.thresholds.get(framing, self.thresholds["_default"])

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CalibrationFile":
        """Load and parse a calibration file.

        Parameters
        ----------
        path:
            Path to a JSON calibration file.  If *None*, the shipped
            ``defaults.json`` is used.
        """
        source = Path(path) if path is not None else _DEFAULTS_PATH
        with source.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        raw_thresholds: dict[str, Any] = raw.get("thresholds", {})
        if "_default" not in raw_thresholds:
            raise ValueError(
                f"calibration file {source} is missing the '_default' framing key"
            )

        parsed: dict[str, FramingThresholds] = {}
        for key, td in raw_thresholds.items():
            if key.startswith("_comment"):
                continue
            parsed[key] = FramingThresholds.from_dict(td)

        return cls(
            fitted=bool(raw.get("fitted", False)),
            fitted_at=raw.get("fitted_at"),
            label_file=raw.get("label_file"),
            thresholds=parsed,
        )


# Module-level singleton loaded from defaults.json.
# Tests may call CalibrationFile.load(custom_path) to override.
_defaults: CalibrationFile | None = None


def load_defaults() -> CalibrationFile:
    """Return the singleton calibration file loaded from defaults.json."""
    global _defaults  # noqa: PLW0603
    if _defaults is None:
        _defaults = CalibrationFile.load()
    return _defaults
