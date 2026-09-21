"""Tests for M2 — presence check.

All tests:
  - use no network and no GPU (AGENTS.md quality bar);
  - use synthetic fixtures and small hand-made images constructed in the test;
  - stub the detector so torch is never imported;
  - are deterministic.

The test suite covers:
  contract.py           — request/verdict JSON round-trip, schema validation,
                          VerdictBuilder logic (accept / reject / unsure).
  calibration/          — load defaults, lookup per framing, fallback to _default.
  checks/presence.py    — all three defect paths, unsure abstain path.
  jury.py               — routing, early stop, check subset filtering.
"""

from __future__ import annotations

import json
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import unittest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _png_1x1(red: int = 0, green: int = 0, blue: int = 0) -> bytes:
    """Produce a valid 1×1 RGB PNG using only the standard library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes((0, red, green, blue)))
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def _write_png(path: Path, *, red: int = 128, green: int = 128, blue: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_1x1(red, green, blue))


def _make_image(tmp: Path, name: str = "frame.png") -> Path:
    p = tmp / name
    _write_png(p)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Stub detector — no torch, no network, deterministic
# ──────────────────────────────────────────────────────────────────────────────

from frame_jury.backends.base import Detection, DetectorBackend  # noqa: E402


class _StubDetector(DetectorBackend):
    """Returns a fixed list of detections regardless of the image."""

    def __init__(self, people: int = 0) -> None:
        self._people = people

    def name(self) -> str:
        return "stub-detector"

    def weights_sha256(self) -> str:
        return "0" * 64

    def detect(self, image_path, *, score_threshold: float = 0.5) -> list[Detection]:
        return [
            Detection(
                label="person",
                confidence=0.9,
                box=(10 + i * 100, 10, 90 + i * 100, 200),
            )
            for i in range(self._people)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Shot / request builders
# ──────────────────────────────────────────────────────────────────────────────

from frame_jury.contract import (  # noqa: E402
    SCHEMA_VERSION,
    ContractError,
    Entity,
    Finding,
    JuryRequest,
    Shot,
    Staging,
    Verdict,
    VerdictBuilder,
)


def _entity(entity_id: str, kind: str = "character") -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "kind": kind,
        "display_name": f"Entity {entity_id}",
        "aliases": [],
        "reference_images": [],
    }


def _make_shot(
    framing: str = "close-up",
    characters: int = 1,
    objects: int = 0,
) -> Shot:
    entities = []
    for i in range(characters):
        entities.append(_entity(f"char-{i:03d}", "character"))
    for i in range(objects):
        entities.append(_entity(f"obj-{i:03d}", "object"))
    return Shot.from_dict(
        {
            "shot_id": "shot-test-001",
            "framing": framing,
            "declared_entities": entities,
            "staging": {
                "purpose": "A test scene.",
                "must_render": [],
                "composition": [],
            },
            "positive_prompt": "test prompt",
            "negative_prompt": "",
        }
    )


def _make_request(tmp: Path, framing: str = "close-up", characters: int = 1) -> JuryRequest:
    image = _make_image(tmp)
    return JuryRequest.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "image_path": str(image),
            "shot": {
                "shot_id": "shot-test-001",
                "framing": framing,
                "declared_entities": [_entity(f"char-{i:03d}") for i in range(characters)],
                "staging": {
                    "purpose": "Test.",
                    "must_render": [],
                    "composition": [],
                },
                "positive_prompt": "test",
                "negative_prompt": "",
            },
            "checks": ["presence"],
            "budget": "cheap",
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Contract tests
# ──────────────────────────────────────────────────────────────────────────────


class TestContractRoundTrip(unittest.TestCase):
    """JSON round-trip for request and verdict."""

    def test_request_round_trip(self) -> None:
        raw = {
            "schema_version": "2.0",
            "image_path": "/tmp/shot.png",
            "shot": {
                "shot_id": "shot-001",
                "framing": "close-up",
                "declared_entities": [
                    {
                        "entity_id": "char-vigia",
                        "kind": "character",
                        "display_name": "O vigia",
                        "aliases": ["guarda-noturno"],
                        "reference_images": [],
                        "visual_identity": "Homem grisalho",
                        "relative_scale": "adulto alto",
                        "approximate_dimensions": "1,85 m",
                    }
                ],
                "staging": {
                    "purpose": "O vigia descobre a anotação.",
                    "must_render": ["O caderno deve estar aberto"],
                    "composition": ["Rosto e caderno em foco"],
                },
                "positive_prompt": "…",
                "negative_prompt": "…",
            },
            "checks": ["presence"],
            "budget": "cheap",
        }
        req = JuryRequest.from_dict(raw)
        self.assertEqual(req.schema_version, "2.0")
        self.assertEqual(req.shot.shot_id, "shot-001")
        self.assertEqual(req.shot.declared_character_count, 1)
        self.assertEqual(req.shot.declared_entities[0].visual_identity, "Homem grisalho")
        # Round-trip through JSON.
        req2 = JuryRequest.from_json(req.to_json())
        self.assertEqual(req2.shot.shot_id, req.shot.shot_id)

    def test_verdict_round_trip(self) -> None:
        v = Verdict(
            schema_version="2.0",
            shot_id="shot-001",
            verdict="reject",
            confidence=0.93,
            findings=[
                Finding(
                    check="presence",
                    defect="duplicated_character",
                    severity="blocking",
                    confidence=0.93,
                    evidence={"people_detected": 2, "declared_characters": 1, "boxes": []},
                    explanation="two people, one declared",
                    entity_id="char-vigia",
                    prompt_hint="add a second person to the negative prompt",
                )
            ],
            measurements={"people": 2},
            timings_ms={"presence": 120.0},
            detectors=[{"check": "presence", "backend": "stub", "weights_sha256": "abc"}],
        )
        v2 = Verdict.from_json(v.to_json())
        self.assertEqual(v2.verdict, "reject")
        self.assertEqual(len(v2.findings), 1)
        self.assertEqual(v2.findings[0].defect, "duplicated_character")
        self.assertEqual(v2.findings[0].prompt_hint, "add a second person to the negative prompt")

    def test_schema_version_mismatch_raises(self) -> None:
        with self.assertRaises(ContractError):
            JuryRequest.from_dict({"schema_version": "1.0", "image_path": "/x"})

    def test_unknown_check_raises(self) -> None:
        with self.assertRaises(ContractError):
            JuryRequest.from_dict(
                {
                    "schema_version": "2.0",
                    "image_path": "/x",
                    "shot": {
                        "shot_id": "s",
                        "framing": "wide",
                        "declared_entities": [],
                        "staging": {"purpose": ".", "must_render": [], "composition": []},
                        "positive_prompt": "x",
                        "negative_prompt": "",
                    },
                    "checks": ["nonexistent_check"],
                    "budget": "cheap",
                }
            )

    def test_unsure_is_a_valid_verdict(self) -> None:
        """unsure is a first-class verdict (SPEC.md §4)."""
        raw = {
            "schema_version": "2.0",
            "shot_id": "shot-001",
            "verdict": "unsure",
            "confidence": 0.5,
        }
        v = Verdict.from_dict(raw)
        self.assertEqual(v.verdict, "unsure")

    def test_prompt_hint_survives_round_trip(self) -> None:
        """prompt_hint is a suggestion, never an edit — it must survive round-trip."""
        f = Finding(
            check="presence",
            defect="missing_entity",
            severity="blocking",
            confidence=0.7,
            evidence={},
            explanation="no person found",
            prompt_hint="state that exactly one person is in frame",
        )
        d = f.to_dict()
        self.assertIn("prompt_hint", d)
        f2 = Finding.from_dict(d)
        self.assertEqual(f2.prompt_hint, "state that exactly one person is in frame")


# ──────────────────────────────────────────────────────────────────────────────
# VerdictBuilder tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVerdictBuilder(unittest.TestCase):
    def test_no_findings_accept(self) -> None:
        b = VerdictBuilder("shot-001")
        v = b.build()
        self.assertEqual(v.verdict, "accept")
        self.assertAlmostEqual(v.confidence, 1.0)

    def test_blocking_finding_rejects(self) -> None:
        b = VerdictBuilder("shot-001")
        b.add_finding(
            Finding(
                check="presence",
                defect="extra_person",
                severity="blocking",
                confidence=0.9,
                evidence={},
                explanation="x",
            )
        )
        v = b.build()
        self.assertEqual(v.verdict, "reject")

    def test_warning_only_gives_unsure(self) -> None:
        """unsure is returned when only warning-severity findings exist."""
        b = VerdictBuilder("shot-001")
        b.add_finding(
            Finding(
                check="presence",
                defect="missing_entity",
                severity="warning",
                confidence=0.6,
                evidence={},
                explanation="abstaining on zero detections",
            )
        )
        v = b.build()
        self.assertEqual(v.verdict, "unsure")

    def test_blocking_beats_warning(self) -> None:
        b = VerdictBuilder("shot-001")
        b.add_finding(
            Finding("presence", "missing_entity", "warning", 0.6, {}, "w")
        )
        b.add_finding(
            Finding("presence", "extra_person", "blocking", 0.9, {}, "b")
        )
        v = b.build()
        self.assertEqual(v.verdict, "reject")

    def test_confidence_is_minimum(self) -> None:
        b = VerdictBuilder("shot-001")
        b.add_finding(Finding("presence", "x", "blocking", 0.8, {}, ""))
        b.add_finding(Finding("presence", "y", "blocking", 0.6, {}, ""))
        v = b.build()
        self.assertAlmostEqual(v.confidence, 0.6)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCalibration(unittest.TestCase):
    def test_defaults_load(self) -> None:
        from frame_jury.calibration.thresholds import CalibrationFile

        cal = CalibrationFile.load()
        self.assertFalse(cal.fitted, "defaults must be marked as unfitted")
        self.assertIn("_default", cal.thresholds)

    def test_framing_lookup_close_up(self) -> None:
        from frame_jury.calibration.thresholds import CalibrationFile

        cal = CalibrationFile.load()
        t = cal.for_framing("close-up")
        self.assertIsInstance(t.person_score_threshold, float)

    def test_framing_lookup_fallback(self) -> None:
        from frame_jury.calibration.thresholds import CalibrationFile

        cal = CalibrationFile.load()
        t = cal.for_framing("portrait")  # not in the file
        self.assertEqual(t, cal.for_framing("_default"))

    def test_custom_calibration_file(self) -> None:
        from frame_jury.calibration.thresholds import CalibrationFile

        data = {
            "fitted": False,
            "fitted_at": None,
            "label_file": None,
            "thresholds": {
                "_default": {
                    "person_score_threshold": 0.42,
                    "missing_entity_abstain_on_empty": False,
                    "extra_person_confidence": 0.7,
                    "duplicated_character_confidence": 0.7,
                    "missing_entity_confidence": 0.7,
                }
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            cal = CalibrationFile.load(tmp_path)
            t = cal.for_framing("_default")
            self.assertAlmostEqual(t.person_score_threshold, 0.42)
            self.assertFalse(t.missing_entity_abstain_on_empty)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_default_key_raises(self) -> None:
        from frame_jury.calibration.thresholds import CalibrationFile

        data = {"fitted": False, "thresholds": {"close-up": {
            "person_score_threshold": 0.5,
            "missing_entity_abstain_on_empty": True,
            "extra_person_confidence": 0.8,
            "duplicated_character_confidence": 0.8,
            "missing_entity_confidence": 0.6,
        }}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                CalibrationFile.load(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Presence check tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPresenceCheck(unittest.TestCase):
    """Tests run with a stub detector — no network, no GPU."""

    def _run(
        self,
        tmp: Path,
        declared_characters: int,
        detected_people: int,
        framing: str = "close-up",
        abstain_on_empty: bool = True,
    ):
        from frame_jury.calibration.thresholds import CalibrationFile
        from frame_jury.checks.presence import run_presence_check

        image = _make_image(tmp)
        shot = _make_shot(framing=framing, characters=declared_characters)
        detector = _StubDetector(people=detected_people)

        # Build a calibration that lets us control abstain_on_empty.
        data = {
            "fitted": False,
            "fitted_at": None,
            "label_file": None,
            "thresholds": {
                "_default": {
                    "person_score_threshold": 0.5,
                    "missing_entity_abstain_on_empty": abstain_on_empty,
                    "extra_person_confidence": 0.80,
                    "duplicated_character_confidence": 0.85,
                    "missing_entity_confidence": 0.65,
                }
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            cal = CalibrationFile.load(tmp_path)
            findings, measurements, elapsed_ms = run_presence_check(
                image, shot, detector, calibration=cal
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        return findings, measurements, elapsed_ms

    def test_correct_count_no_findings(self) -> None:
        """1 declared, 1 detected → no findings (SPEC.md §5)."""
        with tempfile.TemporaryDirectory() as d:
            findings, measurements, _ = self._run(Path(d), declared_characters=1, detected_people=1)
        self.assertEqual(findings, [])
        self.assertEqual(measurements["people_detected"], 1)

    def test_correct_count_two_characters(self) -> None:
        """2 declared, 2 detected → no findings."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(Path(d), declared_characters=2, detected_people=2)
        self.assertEqual(findings, [])

    def test_duplicated_character(self) -> None:
        """1 declared, 2 detected → duplicated_character (SPEC.md §5)."""
        with tempfile.TemporaryDirectory() as d:
            findings, measurements, _ = self._run(Path(d), declared_characters=1, detected_people=2)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].defect, "duplicated_character")
        self.assertEqual(findings[0].severity, "blocking")
        self.assertIn("people_detected", findings[0].evidence)
        self.assertEqual(findings[0].evidence["people_detected"], 2)
        self.assertEqual(findings[0].evidence["declared_characters"], 1)
        # boxes must be present (SPEC.md §4 — every finding carries its evidence)
        self.assertIn("boxes", findings[0].evidence)
        self.assertEqual(len(findings[0].evidence["boxes"]), 2)

    def test_extra_person_no_character_declared(self) -> None:
        """0 declared, 1 detected → extra_person (SPEC.md §5)."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(Path(d), declared_characters=0, detected_people=1)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].defect, "extra_person")
        self.assertEqual(findings[0].severity, "blocking")

    def test_extra_person_multiple(self) -> None:
        """0 declared, 3 detected → extra_person."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(Path(d), declared_characters=0, detected_people=3)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].defect, "extra_person")
        self.assertEqual(findings[0].evidence["people_detected"], 3)

    def test_missing_entity_abstain_by_default(self) -> None:
        """1 declared, 0 detected → unsure (warning severity) when abstain=True."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=0, abstain_on_empty=True
            )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].defect, "missing_entity")
        self.assertEqual(findings[0].severity, "warning")  # not blocking → unsure

    def test_missing_entity_blocking_when_abstain_false(self) -> None:
        """1 declared, 0 detected → reject (blocking severity) when abstain=False."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=0, abstain_on_empty=False
            )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].defect, "missing_entity")
        self.assertEqual(findings[0].severity, "blocking")

    def test_no_character_no_person_clean(self) -> None:
        """0 declared, 0 detected → no findings."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(Path(d), declared_characters=0, detected_people=0)
        self.assertEqual(findings, [])

    def test_findings_carry_evidence(self) -> None:
        """Every finding must carry its evidence numbers (SPEC.md §4)."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(Path(d), declared_characters=1, detected_people=2)
        self.assertTrue(all("people_detected" in f.evidence for f in findings))
        self.assertTrue(all("declared_characters" in f.evidence for f in findings))

    def test_prompt_hint_present(self) -> None:
        """prompt_hint must be set on every finding (SPEC.md §4)."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(Path(d), declared_characters=1, detected_people=2)
        self.assertTrue(all(f.prompt_hint for f in findings))

    def test_elapsed_ms_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _, _, elapsed = self._run(Path(d), declared_characters=1, detected_people=1)
        self.assertGreater(elapsed, 0.0)

    def test_two_declared_zero_detected_two_findings(self) -> None:
        """Two characters declared, zero detected → one finding per character."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _ = self._run(
                Path(d), declared_characters=2, detected_people=0, abstain_on_empty=True
            )
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(f.defect == "missing_entity" for f in findings))
        # Entity ids must differ.
        entity_ids = [f.entity_id for f in findings]
        self.assertEqual(len(set(entity_ids)), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Jury router tests
# ──────────────────────────────────────────────────────────────────────────────


class TestJuryRouter(unittest.TestCase):
    """jury.judge with a stub detector — no torch, no network."""

    def _judge(self, tmp: Path, characters: int, detected_people: int, checks=None):
        from frame_jury.calibration.thresholds import CalibrationFile
        from frame_jury.jury import judge

        data = {
            "fitted": False,
            "fitted_at": None,
            "label_file": None,
            "thresholds": {
                "_default": {
                    "person_score_threshold": 0.5,
                    "missing_entity_abstain_on_empty": True,
                    "extra_person_confidence": 0.80,
                    "duplicated_character_confidence": 0.85,
                    "missing_entity_confidence": 0.65,
                }
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            cal = CalibrationFile.load(tmp_path)
            image = _make_image(tmp)
            request = JuryRequest.from_dict(
                {
                    "schema_version": SCHEMA_VERSION,
                    "image_path": str(image),
                    "shot": {
                        "shot_id": "shot-test-001",
                        "framing": "close-up",
                        "declared_entities": [
                            _entity(f"char-{i:03d}") for i in range(characters)
                        ],
                        "staging": {
                            "purpose": "Test.",
                            "must_render": [],
                            "composition": [],
                        },
                        "positive_prompt": "test",
                        "negative_prompt": "",
                    },
                    "checks": checks or ["presence"],
                    "budget": "cheap",
                }
            )
            return judge(request, detector=_StubDetector(people=detected_people), calibration=cal)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_accept_correct_count(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=1)
        self.assertEqual(verdict.verdict, "accept")

    def test_reject_duplicated_character(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=2)
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.findings), 1)
        self.assertEqual(verdict.findings[0].defect, "duplicated_character")

    def test_unsure_missing_entity_default_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=0)
        self.assertEqual(verdict.verdict, "unsure")

    def test_verdict_has_timing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=1)
        self.assertIn("presence", verdict.timings_ms)
        self.assertGreaterEqual(verdict.timings_ms["presence"], 0.0)

    def test_verdict_has_detectors_list(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=1)
        self.assertEqual(len(verdict.detectors), 1)
        self.assertEqual(verdict.detectors[0]["backend"], "stub-detector")

    def test_verdict_has_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=2)
        self.assertIn("people_detected", verdict.measurements)
        self.assertEqual(verdict.measurements["people_detected"], 2)

    def test_early_stop_on_blocking(self) -> None:
        """budget=cheap: only one check runs; early stop on blocking finding."""
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=2, checks=["presence"])
        # Only presence ran (identity is M3); verdict is reject.
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.timings_ms), 1)

    def test_extra_person_reject(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=0, detected_people=1)
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(verdict.findings[0].defect, "extra_person")

    def test_schema_version_in_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=1)
        self.assertEqual(verdict.schema_version, SCHEMA_VERSION)

    def test_shot_id_in_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=1)
        self.assertEqual(verdict.shot_id, "shot-test-001")

    def test_verdict_to_json_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=2)
        raw = verdict.to_json()
        parsed = json.loads(raw)
        self.assertEqual(parsed["schema_version"], "2.0")
        self.assertIn("findings", parsed)

    def test_determinism(self) -> None:
        """Same image and declaration give the same verdict (SPEC.md §4)."""
        with tempfile.TemporaryDirectory() as d:
            v1 = self._judge(Path(d), characters=1, detected_people=2)
            v2 = self._judge(Path(d), characters=1, detected_people=2)
        self.assertEqual(v1.verdict, v2.verdict)
        self.assertEqual(v1.to_json(), v2.to_json())


# ──────────────────────────────────────────────────────────────────────────────
# Licensing boundary test
# ──────────────────────────────────────────────────────────────────────────────


class TestLicensingBoundary(unittest.TestCase):
    """frame_jury must not import any competitor package (AGENTS.md §licensing)."""

    _BANNED = ["ultralytics", "insightface", "deepface"]

    def test_no_competitor_imports(self) -> None:
        import sys

        for banned in self._BANNED:
            self.assertNotIn(
                banned,
                sys.modules,
                f"competitor package {banned!r} was imported by frame_jury",
            )


if __name__ == "__main__":
    unittest.main()
