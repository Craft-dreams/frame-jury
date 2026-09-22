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
    Abstention,
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
    other_people_allowed: bool = True,
    expected_people_count: int | None = None,
) -> Shot:
    entities = []
    for i in range(characters):
        entities.append(_entity(f"char-{i:03d}", "character"))
    for i in range(objects):
        entities.append(_entity(f"obj-{i:03d}", "object"))
    raw = {
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
        "other_people_allowed": other_people_allowed,
    }
    if expected_people_count is not None:
        raw["expected_people_count"] = expected_people_count
    return Shot.from_dict(raw)


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

    def test_other_people_allowed_default_is_true(self) -> None:
        """A request dict without other_people_allowed parses to True."""
        raw = {
            "schema_version": "2.0",
            "image_path": "/tmp/shot.png",
            "shot": {
                "shot_id": "shot-001",
                "framing": "close-up",
                "declared_entities": [],
                "staging": {"purpose": "p", "must_render": [], "composition": []},
                "positive_prompt": "p",
                "negative_prompt": "",
            },
        }
        req = JuryRequest.from_dict(raw)
        self.assertTrue(req.shot.other_people_allowed)

    def test_other_people_allowed_false_round_trip(self) -> None:
        """{"other_people_allowed": false} parses to False and survives a to_dict/from_dict round trip."""
        raw = {
            "schema_version": "2.0",
            "image_path": "/tmp/shot.png",
            "shot": {
                "shot_id": "shot-001",
                "framing": "close-up",
                "declared_entities": [],
                "staging": {"purpose": "p", "must_render": [], "composition": []},
                "positive_prompt": "p",
                "negative_prompt": "",
                "other_people_allowed": False,
            },
        }
        req = JuryRequest.from_dict(raw)
        self.assertFalse(req.shot.other_people_allowed)
        d = req.shot.to_dict()
        self.assertIn("other_people_allowed", d)
        self.assertIs(d["other_people_allowed"], False)
        shot2 = Shot.from_dict(d)
        self.assertFalse(shot2.other_people_allowed)

        # Also full JuryRequest round-trip:
        req2 = JuryRequest.from_dict(req.to_dict())
        self.assertFalse(req2.shot.other_people_allowed)

    def test_other_people_allowed_malformed_raises_contract_error(self) -> None:
        """{"other_people_allowed": "false"} and {"other_people_allowed": 0} raise ContractError."""
        for malformed in ["false", 0, 1, None, []]:
            raw = {
                "schema_version": "2.0",
                "image_path": "/tmp/shot.png",
                "shot": {
                    "shot_id": "shot-001",
                    "framing": "close-up",
                    "declared_entities": [],
                    "staging": {"purpose": "p", "must_render": [], "composition": []},
                    "positive_prompt": "p",
                    "negative_prompt": "",
                    "other_people_allowed": malformed,
                },
            }
            with self.assertRaises(ContractError) as ctx:
                JuryRequest.from_dict(raw)
            self.assertIn("other_people_allowed", str(ctx.exception))

    def test_negative_expected_people_count_raises_contract_error(self) -> None:
        raw = {
            "shot_id": "shot-001",
            "framing": "close-up",
            "declared_entities": [],
            "staging": {"purpose": "p", "must_render": [], "composition": []},
            "positive_prompt": "p",
            "negative_prompt": "",
            "expected_people_count": -1,
        }
        with self.assertRaises(ContractError) as ctx:
            Shot.from_dict(raw)
        self.assertIn("expected_people_count", str(ctx.exception))



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

    def test_warning_only_gives_accept(self) -> None:
        """A warning finding on its own produces accept with warnings (SPEC.md §4)."""
        b = VerdictBuilder("shot-001")
        b.add_finding(
            Finding(
                check="presence",
                defect="missing_entity",
                severity="warning",
                confidence=0.6,
                evidence={},
                explanation="non-blocking issue",
            )
        )
        v = b.build()
        self.assertEqual(v.verdict, "accept")
        self.assertEqual(len(v.findings), 1)

    def test_abstention_only_gives_unsure(self) -> None:
        """unsure is returned when only abstentions exist (SPEC.md §4)."""
        b = VerdictBuilder("shot-001")
        b.add_abstention(
            Abstention(
                check="presence",
                reason="no_person_in_frame",
                entity_id="char-001",
                explanation="abstaining on zero detections",
            )
        )
        v = b.build()
        self.assertEqual(v.verdict, "unsure")
        self.assertEqual(len(v.abstentions), 1)

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
        other_people_allowed: bool = True,
        expected_people_count: int | None = None,
    ):
        from frame_jury.calibration.thresholds import CalibrationFile
        from frame_jury.checks.presence import run_presence_check

        image = _make_image(tmp)
        shot = _make_shot(
            framing=framing,
            characters=declared_characters,
            other_people_allowed=other_people_allowed,
            expected_people_count=expected_people_count,
        )
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
            findings, abstentions, measurements, elapsed_ms = run_presence_check(
                image, shot, detector, calibration=cal
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        return findings, abstentions, measurements, elapsed_ms

    def test_explicit_zero_rejects_detected_person_when_others_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d),
                declared_characters=1,
                detected_people=1,
                other_people_allowed=False,
                expected_people_count=0,
            )
        self.assertEqual([finding.defect for finding in findings], ["extra_person"])
        self.assertEqual(abstentions, [])
        self.assertEqual(measurements["expected_people_count"], 0)

    def test_explicit_one_with_one_detected_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d),
                declared_characters=0,
                detected_people=1,
                other_people_allowed=False,
                expected_people_count=1,
            )
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])
        self.assertEqual(measurements["expected_people_count"], 1)

    def test_collective_entity_explicit_two_with_two_detected_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(
                Path(d),
                declared_characters=1,
                detected_people=2,
                other_people_allowed=False,
                expected_people_count=2,
            )
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])

    def test_collective_entity_explicit_two_rejects_three_detected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(
                Path(d),
                declared_characters=1,
                detected_people=3,
                other_people_allowed=False,
                expected_people_count=2,
            )
        self.assertEqual([finding.defect for finding in findings], ["extra_person"])
        self.assertEqual(findings[0].evidence["expected_people_count"], 2)
        self.assertEqual(abstentions, [])

    def test_explicit_two_reports_one_detected_as_missing_entity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(
                Path(d),
                declared_characters=1,
                detected_people=1,
                expected_people_count=2,
            )
        self.assertEqual([finding.defect for finding in findings], ["missing_entity"])
        self.assertEqual(findings[0].evidence["shortage"], 1)
        self.assertEqual(abstentions, [])

    def test_correct_count_no_findings(self) -> None:
        """1 declared, 1 detected → no findings (SPEC.md §5)."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(Path(d), declared_characters=1, detected_people=1)
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])
        self.assertEqual(measurements["people_detected"], 1)

    def test_correct_count_two_characters(self) -> None:
        """2 declared, 2 detected → no findings."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(Path(d), declared_characters=2, detected_people=2)
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])

    def test_background_people_allowed_by_default_one_declared(self) -> None:
        """1 declared, 2 detected, other_people_allowed default → no findings, background_people == 1."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d), declared_characters=1, detected_people=2
            )
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])
        self.assertEqual(measurements["background_people"], 1)
        self.assertTrue(measurements["other_people_allowed"])

    def test_establishing_street_shot_background_people(self) -> None:
        """0 declared, 3 detected, default → no findings (an establishing street shot), background_people == 3."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d), declared_characters=0, detected_people=3
            )
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])
        self.assertEqual(measurements["background_people"], 3)
        self.assertTrue(measurements["other_people_allowed"])

    def test_extra_person_one_declared_forbidden(self) -> None:
        """1 declared, 2 detected, other_people_allowed=False → exactly one extra_person, surplus == 1, no entity_id."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d), declared_characters=1, detected_people=2, other_people_allowed=False
            )
        self.assertEqual(len(findings), 1)
        self.assertEqual(abstentions, [])
        f = findings[0]
        self.assertEqual(f.defect, "extra_person")
        self.assertEqual(f.severity, "blocking")
        self.assertEqual(f.evidence["surplus"], 1)
        self.assertEqual(f.evidence["people_detected"], 2)
        self.assertEqual(f.evidence["declared_characters"], 1)
        self.assertFalse(f.evidence["other_people_allowed"])
        self.assertIn("boxes", f.evidence)
        self.assertEqual(f.entity_id, "")
        self.assertFalse(measurements["other_people_allowed"])

    def test_extra_person_zero_declared_forbidden(self) -> None:
        """0 declared, 1 detected, other_people_allowed=False → exactly one extra_person."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d), declared_characters=0, detected_people=1, other_people_allowed=False
            )
        self.assertEqual(len(findings), 1)
        self.assertEqual(abstentions, [])
        f = findings[0]
        self.assertEqual(f.defect, "extra_person")
        self.assertEqual(f.severity, "blocking")
        self.assertEqual(f.evidence["surplus"], 1)
        self.assertEqual(f.entity_id, "")

    def test_correct_count_other_people_forbidden(self) -> None:
        """2 declared, 2 detected, other_people_allowed=False → no findings."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, measurements, _ = self._run(
                Path(d), declared_characters=2, detected_people=2, other_people_allowed=False
            )
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])
        self.assertNotIn("background_people", measurements)
        self.assertFalse(measurements["other_people_allowed"])

    def test_presence_never_emits_duplicated_character(self) -> None:
        """No test anywhere produces duplicated_character from presence: assert it for 1-declared/2-detected under both values of the flag."""
        with tempfile.TemporaryDirectory() as d:
            findings_allowed, _, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=2, other_people_allowed=True
            )
            findings_forbidden, _, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=2, other_people_allowed=False
            )
        self.assertFalse(any(f.defect == "duplicated_character" for f in findings_allowed))
        self.assertFalse(any(f.defect == "duplicated_character" for f in findings_forbidden))

    def test_missing_entity_abstain_by_default(self) -> None:
        """1 declared, 0 detected → unsure (first-class Abstention) when abstain=True."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=0, abstain_on_empty=True
            )
        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "presence")
        self.assertEqual(abstentions[0].reason, "no_person_in_frame")
        self.assertEqual(abstentions[0].entity_id, "char-000")

    def test_missing_entity_blocking_when_abstain_false(self) -> None:
        """1 declared, 0 detected → reject (blocking severity) when abstain=False."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=0, abstain_on_empty=False
            )
        self.assertEqual(len(findings), 1)
        self.assertEqual(abstentions, [])
        self.assertEqual(findings[0].defect, "missing_entity")
        self.assertEqual(findings[0].severity, "blocking")

    def test_no_character_no_person_clean(self) -> None:
        """0 declared, 0 detected → no findings."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(Path(d), declared_characters=0, detected_people=0)
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])

    def test_findings_carry_evidence(self) -> None:
        """Every finding must carry its evidence numbers (SPEC.md §4)."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=2, other_people_allowed=False
            )
        self.assertEqual(len(findings), 1)
        self.assertTrue(all("people_detected" in f.evidence for f in findings))
        self.assertTrue(all("declared_characters" in f.evidence for f in findings))
        self.assertTrue(all("surplus" in f.evidence for f in findings))

    def test_prompt_hint_present(self) -> None:
        """prompt_hint must be set on every finding (SPEC.md §4)."""
        with tempfile.TemporaryDirectory() as d:
            findings, _, _, _ = self._run(
                Path(d), declared_characters=1, detected_people=2, other_people_allowed=False
            )
        self.assertEqual(len(findings), 1)
        self.assertTrue(all(f.prompt_hint for f in findings))

    def test_elapsed_ms_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, elapsed = self._run(Path(d), declared_characters=1, detected_people=1)
        self.assertGreater(elapsed, 0.0)

    def test_two_declared_zero_detected_two_abstentions(self) -> None:
        """Two characters declared, zero detected → one abstention per character."""
        with tempfile.TemporaryDirectory() as d:
            findings, abstentions, _, _ = self._run(
                Path(d), declared_characters=2, detected_people=0, abstain_on_empty=True
            )
        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 2)
        self.assertTrue(all(a.check == "presence" for a in abstentions))
        self.assertTrue(all(a.reason == "no_person_in_frame" for a in abstentions))
        # Entity ids must differ.
        entity_ids = [a.entity_id for a in abstentions]
        self.assertEqual(len(set(entity_ids)), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Jury router tests
# ──────────────────────────────────────────────────────────────────────────────


class TestJuryRouter(unittest.TestCase):
    """jury.judge with a stub detector — no torch, no network."""

    def _judge(
        self,
        tmp: Path,
        characters: int,
        detected_people: int,
        checks=None,
        other_people_allowed: bool = True,
    ):
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
                        "other_people_allowed": other_people_allowed,
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

    def test_reject_extra_person_when_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=1, detected_people=2, other_people_allowed=False)
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.findings), 1)
        self.assertEqual(verdict.findings[0].defect, "extra_person")

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
        self.assertIn("background_people", verdict.measurements)
        self.assertEqual(verdict.measurements["background_people"], 1)
        self.assertTrue(verdict.measurements["other_people_allowed"])

    def test_early_stop_on_blocking(self) -> None:
        """budget=cheap: only one check runs; early stop on blocking finding."""
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(
                Path(d), characters=1, detected_people=2, checks=["presence"], other_people_allowed=False
            )
        # Only presence ran (identity is M3); verdict is reject.
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.timings_ms), 1)

    def test_extra_person_reject(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = self._judge(Path(d), characters=0, detected_people=1, other_people_allowed=False)
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
