"""tests.test_harness — tests for harness, scoring, competitor interfaces, and leaderboard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmarks.competitors import (
    CompetitorResult,
    FrameJuryCompetitor,
    NullCompetitor,
    _defect_for_abstention,
)
from benchmarks.corpus.schema import write_cases
from benchmarks.run import (
    get_pinned_versions,
    render_leaderboard,
    run_benchmark,
)
from benchmarks.scoring import score
from frame_jury.backends.base import Detection, DetectorBackend, FaceBackend
from frame_jury.calibration.thresholds import CalibrationFile
from frame_jury.contract import Abstention
from tests.helpers import make_case


class StubDetector(DetectorBackend):
    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []

    def name(self) -> str:
        return "stub-detector"

    def weights_sha256(self) -> str:
        return "0" * 64

    def detect(self, image_path: str | Path, *, score_threshold: float = 0.5) -> list[Detection]:
        return self._detections


class StubFaceBackend(FaceBackend):
    def name(self) -> str:
        return "stub-face"

    def weights_sha256(self) -> str:
        return "0" * 64

    def detect_faces(self, image_path: str | Path) -> list[Detection]:
        return []

    def embed(
        self,
        image_path: str | Path,
        box: tuple[int, int, int, int],
    ) -> list[float]:
        return [0.0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 1.0


class HarnessTests(unittest.TestCase):
    def test_scoring_maths_hand_calculated(self) -> None:
        """Precision, recall, and F1 match hand-calculated values on known TP/FP/FN."""
        # 4 cases:
        # c1: GT pos, Pred pos -> TP
        # c2: GT pos, Pred pos -> TP
        # c3: GT neg, Pred pos -> FP
        # c4: GT pos, Pred neg -> FN
        defect = "duplicated_character"
        results = {
            "c1": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c2": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c3": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c4": CompetitorResult(defects=frozenset(), abstained=frozenset()),
        }
        labels = {
            "c1": {defect},
            "c2": {defect},
            "c3": {"clean"},
            "c4": {defect},
        }

        s = score(results, labels, defect)

        # TP = 2, FP = 1, FN = 1
        # Precision = 2 / (2 + 1) = 2/3
        # Recall = 2 / (2 + 1) = 2/3
        # F1 = 2 * (2/3) * (2/3) / ((2/3) + (2/3)) = 2/3
        self.assertAlmostEqual(s["precision"], 2 / 3)
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["f1"], 2 / 3)
        self.assertEqual(s["support"], 3.0)
        self.assertEqual(s["tp"], 2.0)
        self.assertEqual(s["fp"], 1.0)
        self.assertEqual(s["fn"], 1.0)
        self.assertEqual(s["abstain_rate"], 0.0)

    def test_zero_denominator_metric_is_none_not_zero(self) -> None:
        """A metric whose denominator is zero returns None, never 0.0."""
        defect = "wrong_identity"

        # Case A: TP=0, FP=0 -> precision denominator is zero
        results_a = {
            "c1": CompetitorResult(defects=frozenset(), abstained=frozenset()),
        }
        labels_a = {
            "c1": {defect},  # FN=1
        }
        s_a = score(results_a, labels_a, defect)
        self.assertIsNone(s_a["precision"])
        self.assertEqual(s_a["recall"], 0.0)
        self.assertIsNone(s_a["f1"])  # P is None -> F1 is None

        # Case B: TP=0, FN=0 -> recall denominator is zero
        results_b = {
            "c1": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
        }
        labels_b = {
            "c1": {"clean"},  # FP=1
        }
        s_b = score(results_b, labels_b, defect)
        self.assertEqual(s_b["precision"], 0.0)
        self.assertIsNone(s_b["recall"])
        self.assertIsNone(s_b["f1"])  # R is None -> F1 is None

        # Case C: TP=0, FP=1, FN=1 -> precision=0, recall=0, P+R=0 -> F1 denominator is zero
        results_c = {
            "c1": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c2": CompetitorResult(defects=frozenset(), abstained=frozenset()),
        }
        labels_c = {
            "c1": {"clean"},
            "c2": {defect},
        }
        s_c = score(results_c, labels_c, defect)
        self.assertEqual(s_c["precision"], 0.0)
        self.assertEqual(s_c["recall"], 0.0)
        self.assertIsNone(s_c["f1"], "F1 must be None when Precision + Recall is 0.0")

    def test_uncertain_cases_excluded_from_scoring(self) -> None:
        """Uncertain cases are excluded entirely and never guessed."""
        defect = "extra_person"
        results = {
            "c1": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c2": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
        }
        labels = {
            "c1": {defect},
            "c2": {"uncertain"},  # must be ignored entirely
        }
        s = score(results, labels, defect)
        self.assertEqual(s["precision"], 1.0)
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["fp"], 0.0)
        self.assertEqual(s["support"], 1.0)

    def test_legacy_broken_anatomy_excluded_from_body_and_hands(self) -> None:
        """Legacy schema-1.0 broken_anatomy is excluded from broken_body and broken_hands."""
        results = {
            "c1": CompetitorResult(
                defects=frozenset(["broken_body", "broken_hands", "extra_person"]),
                abstained=frozenset(),
            ),
        }
        labels = {
            "c1": {"broken_anatomy"},
        }

        # Excluded from broken_body
        s_body = score(results, labels, "broken_body")
        self.assertIsNone(s_body["precision"])
        self.assertIsNone(s_body["recall"])
        self.assertEqual(s_body["support"], 0.0)
        self.assertEqual(s_body["excluded_legacy"], 1.0)

        # Excluded from broken_hands
        s_hands = score(results, labels, "broken_hands")
        self.assertIsNone(s_hands["precision"])
        self.assertIsNone(s_hands["recall"])
        self.assertEqual(s_hands["support"], 0.0)
        self.assertEqual(s_hands["excluded_legacy"], 1.0)

        # NOT excluded from other defects (e.g. extra_person)
        s_extra = score(results, labels, "extra_person")
        self.assertEqual(s_extra["excluded_legacy"], 0.0)
        self.assertEqual(s_extra["fp"], 1.0)

    def test_abstained_case_excluded_from_precision_and_recall_and_counted_in_rate(self) -> None:
        """Abstentions are neither positive nor negative; excluded from P/R and in abstain rate."""
        defect = "wrong_identity"
        results = {
            "c1": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c2": CompetitorResult(defects=frozenset(), abstained=frozenset([defect])),
        }
        labels = {
            "c1": {defect},
            "c2": {defect},
        }
        s = score(results, labels, defect)
        # c2 abstained: excluded from recall denominator.
        # c1 is TP.
        self.assertEqual(s["precision"], 1.0)
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["support"], 1.0)
        self.assertEqual(s["fn"], 0.0)
        self.assertEqual(s["abstain_rate"], 0.5)

    def test_null_competitor_scores_zero_recall_and_no_false_positives(self) -> None:
        """NullCompetitor scores 0.0 recall and 0 false positives on positive ground truth."""
        defect = "duplicated_character"
        comp = NullCompetitor()
        self.assertEqual(comp.name, "null")
        self.assertEqual(comp.licence, "n/a")

        cases = [
            {"case_id": "c1", "image_path": "x.png"},
            {"case_id": "c2", "image_path": "x.png"},
            {"case_id": "c3", "image_path": "x.png"},
        ]
        results = {c["case_id"]: comp.judge_case(c) for c in cases}
        labels = {
            "c1": {defect},
            "c2": {defect},
            "c3": {"clean"},
        }
        s = score(results, labels, defect)
        self.assertEqual(s["recall"], 0.0)
        self.assertEqual(s["fp"], 0.0)
        self.assertIsNone(s["precision"])
        self.assertIsNone(s["f1"])
        self.assertEqual(s["abstain_rate"], 0.0)

    def test_zero_label_path_writes_table_with_dashes_and_does_not_raise(self) -> None:
        """Zero-label path writes a valid table with '—' metrics and does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases_path = root / "cases.jsonl"
            out_path = root / "LEADERBOARD.md"

            # Create a 1x1 image so case passes existence check
            img_path = root / "shot-001.png"
            from tests.helpers import write_png

            write_png(img_path)

            case = make_case(root, "run-1", "shot-001")
            case["image_path"] = str(img_path.resolve())
            write_cases(cases_path, [case])

            content = run_benchmark(
                cases_path=cases_path,
                labels_path=root / "nonexistent-labels.jsonl",
                out_path=out_path,
                runs=1,
                competitors=[NullCompetitor()],
                timestamp="2026-09-21T22:00:00Z",
            )

            self.assertTrue(out_path.exists())
            self.assertIn("- **Labelled**: 0", content)
            self.assertIn("| null |", content)
            self.assertIn("—", content)
            self.assertIn("*peak RAM (py) measures Python allocations only", content)

    def test_leaderboard_renderer_deterministic(self) -> None:
        """Leaderboard renderer produces identical bytes for identical inputs regardless of row order."""
        pinned = get_pinned_versions()
        timestamp = "2026-09-21T12:00:00Z"
        row1 = {
            "system": "null",
            "defect": "duplicated_character",
            "precision": None,
            "recall": 0.0,
            "f1": None,
            "abstain_pct": 0.0,
            "ms_per_frame": 0.005,
            "peak_ram_mb": 0.123,
            "licence": "n/a",
        }
        row2 = {
            "system": "frame-jury",
            "defect": "duplicated_character",
            "precision": 0.95,
            "recall": 0.92,
            "f1": 0.935,
            "abstain_pct": 5.0,
            "ms_per_frame": 12.345,
            "peak_ram_mb": 1.456,
            "licence": "AGPL-3.0-or-later",
        }

        # Order [row1, row2]
        content_a = render_leaderboard(
            split_name="test",
            case_count=10,
            labelled_count=5,
            excluded_legacy_count=0,
            timestamp=timestamp,
            pinned_versions=pinned,
            rows=[row1, row2],
        )

        # Order [row2, row1]
        content_b = render_leaderboard(
            split_name="test",
            case_count=10,
            labelled_count=5,
            excluded_legacy_count=0,
            timestamp=timestamp,
            pinned_versions=pinned,
            rows=[row2, row1],
        )

        self.assertEqual(content_a, content_b, "Renderer must sort rows deterministically")
        # System 'frame-jury' comes before 'null' alphabetically
        pos_fj = content_a.index("| frame-jury |")
        pos_null = content_a.index("| null |")
        self.assertLess(pos_fj, pos_null)

    def test_frame_jury_competitor_with_stubs(self) -> None:
        """FrameJuryCompetitor builds JuryRequest, calls judge, and maps findings and abstentions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_path = root / "frame.png"
            from tests.helpers import write_png

            write_png(img_path)

            case = make_case(root, "run-test", "shot-test")
            case["image_path"] = str(img_path.resolve())
            case["shot"]["declared_entities"] = [
                {
                    "entity_id": "char-1",
                    "kind": "character",
                    "display_name": "Test Char",
                    "aliases": [],
                    "reference_images": [],
                }
            ]

            from frame_jury.calibration.thresholds import load_defaults

            comp = FrameJuryCompetitor(
                detector=StubDetector([]),
                face_backend=StubFaceBackend(),
                calibration=load_defaults(),
            )
            self.assertEqual(comp.name, "frame-jury")
            self.assertEqual(comp.licence, "AGPL-3.0-or-later")

            result = comp.judge_case(case)
            self.assertIsInstance(result, CompetitorResult)
            # Declared 1 character, 0 detected, missing_entity_abstain_on_empty=True
            # -> emits Abstention for missing_entity and identity emits Abstention for no_reference_image
            self.assertIn("missing_entity", result.abstained)
            self.assertIn("wrong_identity", result.abstained)
            self.assertEqual(len(result.defects), 0)

    def test_abstain_rate_independent_of_labels(self) -> None:
        """abstain_rate is computed over all results regardless of whether labels exist."""
        defect = "wrong_identity"
        results = {
            "c1": CompetitorResult(defects=frozenset(), abstained=frozenset([defect])),
            "c2": CompetitorResult(defects=frozenset(), abstained=frozenset()),
            "c3": CompetitorResult(defects=frozenset([defect]), abstained=frozenset()),
            "c4": CompetitorResult(defects=frozenset(), abstained=frozenset()),
        }

        # Scored with no labels -> abstain_rate is 1/4 = 0.25
        s_no_labels = score(results, {}, defect)
        self.assertEqual(s_no_labels["abstain_rate"], 0.25)
        self.assertIsNone(s_no_labels["precision"])
        self.assertIsNone(s_no_labels["recall"])

        # Scored with labels on 2 of the cases -> abstain_rate is still 1/4 = 0.25
        labels = {
            "c1": {defect},
            "c2": {"clean"},
        }
        s_with_labels = score(results, labels, defect)
        self.assertEqual(s_with_labels["abstain_rate"], 0.25)

    def test_defect_for_abstention_mapping(self) -> None:
        """_defect_for_abstention maps known checks and returns None for unknown checks."""
        a_id = Abstention(check="identity", reason="no_face_in_frame")
        self.assertEqual(_defect_for_abstention(a_id), "wrong_identity")

        a_pres = Abstention(check="presence", reason="no_person_in_frame")
        self.assertEqual(_defect_for_abstention(a_pres), "missing_entity")

        # Unknown check maps to None
        a_unknown = Abstention(check="unknown_check", reason="x")
        self.assertIsNone(_defect_for_abstention(a_unknown))

    def test_unknown_abstention_does_not_appear_in_competitor_abstained(self) -> None:
        """An Abstention(check='unknown_check', reason='x') maps to None and does not appear in CompetitorResult.abstained."""
        from unittest.mock import patch
        from frame_jury.contract import Verdict

        comp = FrameJuryCompetitor(
            detector=StubDetector([]),
            face_backend=StubFaceBackend(),
        )
        fake_verdict = Verdict(
            schema_version="2.0",
            shot_id="shot-1",
            verdict="accept",
            confidence=1.0,
            findings=[],
            abstentions=[
                Abstention(check="unknown_check", reason="x"),
                Abstention(check="identity", reason="no_face_in_frame"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            case = make_case(Path(tmpdir), "run-test", "shot-test")
            with patch("benchmarks.competitors.judge", return_value=fake_verdict):
                res = comp.judge_case(case)
                self.assertEqual(res.abstained, frozenset(["wrong_identity"]))
                self.assertNotIn("unknown_check", res.abstained)
