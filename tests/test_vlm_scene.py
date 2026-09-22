"""tests.test_vlm_scene — unit tests for the VLM scene check and scorer backend port."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from frame_jury.backends.base import VlmScorerBackend
from frame_jury.backends.vlm_qwen3 import MODEL_ID, MODEL_REVISION, Qwen3VlmScorer
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.checks.vlm_scene import (
    build_duplicated_character_question,
    build_missing_entity_question,
    run_vlm_scene_check,
)
from frame_jury.contract import (
    DEFECT_DUPLICATED_CHARACTER,
    DEFECT_MISSING_ENTITY,
    Entity,
    JuryRequest,
    Shot,
    Staging,
    Verdict,
)
from frame_jury.jury import judge
from tests.helpers import write_png


class StubVlmScorer(VlmScorerBackend):
    """Deterministic stub scorer for offline testing without GPU or model weights."""

    def __init__(self, scores: dict[str, float] | float = 0.0) -> None:
        self.scores = scores
        self.call_count = 0
        self.calls: list[tuple[str | Path, str]] = []

    def name(self) -> str:
        return "stub-qwen3-vl-8b"

    def revision(self) -> str:
        return "stub-rev-42"

    def score_yes_no(self, image_path: str | Path, question: str) -> float:
        self.call_count += 1
        self.calls.append((image_path, question))
        if isinstance(self.scores, dict):
            for pattern, score in self.scores.items():
                if pattern in question:
                    return score
            return 0.0
        return float(self.scores)


def _make_shot(
    *,
    shot_id: str = "shot-001",
    framing: str = "close-up",
    characters: list[str] | None = None,
    objects: list[str] | None = None,
) -> Shot:
    entities: list[Entity] = []
    if characters:
        for idx, name in enumerate(characters):
            entities.append(
                Entity(
                    entity_id=f"char-{idx}",
                    kind="character",
                    display_name=name,
                    aliases=(),
                    reference_images=(),
                )
            )
    if objects:
        for idx, name in enumerate(objects):
            entities.append(
                Entity(
                    entity_id=f"obj-{idx}",
                    kind="object",
                    display_name=name,
                    aliases=(),
                    reference_images=(),
                )
            )
    return Shot(
        shot_id=shot_id,
        framing=framing,
        declared_entities=tuple(entities),
        staging=Staging(purpose="test purpose", must_render=(), composition=()),
        positive_prompt="test",
        negative_prompt="",
    )


class TestVlmSceneCheck(unittest.TestCase):
    """Unit tests for frame_jury.checks.vlm_scene and judge integration."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.image_path = self.tmp / "frame.png"
        write_png(self.image_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_duplicated_character_question_contains_every_declared_character_name(self) -> None:
        """The question for duplicated_character must contain every declared character's name."""
        char_names = ["Captain Nemo", "Professor Aronnax", "Ned Land"]
        shot = _make_shot(characters=char_names, objects=["Submarine Nautilus"])

        question = build_duplicated_character_question(shot)

        for name in char_names:
            self.assertIn(
                name,
                question,
                f"Declared character '{name}' was not found in the duplicated_character question.",
            )
        self.assertIn("Submarine Nautilus", question)

    def test_missing_entity_question_contains_declared_entities(self) -> None:
        """The question for missing_entity must contain all declared entities."""
        shot = _make_shot(characters=["Alice"], objects=["Magic Key"])
        question = build_missing_entity_question(shot)
        self.assertIn("Alice", question)
        self.assertIn("Magic Key", question)

    def test_findings_above_threshold(self) -> None:
        """Scores above threshold emit blocking findings with score as confidence and full evidence."""
        shot = _make_shot(characters=["Hero"], objects=[])
        # Stub returns 0.85 for duplicate and 0.75 for missing entity (default threshold is 0.5)
        scorer = StubVlmScorer(
            scores={
                "Does any ONE of": 0.85,
                "missing from the image": 0.75,
            }
        )

        request = JuryRequest(
            schema_version="2.0",
            image_path=str(self.image_path),
            shot=shot,
            checks=("vlm_scene",),
            budget="full",
        )
        verdict = judge(request, vlm_scorer=scorer)

        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.findings), 2)

        defects = {f.defect: f for f in verdict.findings}
        self.assertIn(DEFECT_DUPLICATED_CHARACTER, defects)
        self.assertIn(DEFECT_MISSING_ENTITY, defects)

        dup_finding = defects[DEFECT_DUPLICATED_CHARACTER]
        self.assertEqual(dup_finding.check, "vlm_scene")
        self.assertEqual(dup_finding.severity, "blocking")
        self.assertAlmostEqual(dup_finding.confidence, 0.85)
        self.assertEqual(dup_finding.evidence["model"], "stub-qwen3-vl-8b")
        self.assertEqual(dup_finding.evidence["revision"], "stub-rev-42")
        self.assertAlmostEqual(dup_finding.evidence["p_yes"], 0.85)
        self.assertIn("Hero", dup_finding.evidence["question"])

        miss_finding = defects[DEFECT_MISSING_ENTITY]
        self.assertEqual(miss_finding.check, "vlm_scene")
        self.assertEqual(miss_finding.severity, "blocking")
        self.assertAlmostEqual(miss_finding.confidence, 0.75)
        self.assertEqual(miss_finding.evidence["model"], "stub-qwen3-vl-8b")
        self.assertEqual(miss_finding.evidence["revision"], "stub-rev-42")
        self.assertAlmostEqual(miss_finding.evidence["p_yes"], 0.75)

    def test_findings_below_threshold_accepts(self) -> None:
        """Scores below or equal to threshold do not emit findings and result in accept."""
        shot = _make_shot(characters=["Hero"])
        scorer = StubVlmScorer(scores=0.20)  # below default 0.50

        request = JuryRequest(
            schema_version="2.0",
            image_path=str(self.image_path),
            shot=shot,
            checks=("vlm_scene",),
            budget="full",
        )
        verdict = judge(request, vlm_scorer=scorer)

        self.assertEqual(verdict.verdict, "accept")
        self.assertEqual(len(verdict.findings), 0)
        self.assertEqual(len(verdict.abstentions), 0)
        self.assertEqual(scorer.call_count, 2)

    def test_abstention_when_backend_missing(self) -> None:
        """When budget='full' and VLM backend is unavailable, emits an abstention (unsure, not silent pass)."""
        shot = _make_shot(characters=["Hero"])

        findings, abstentions, measurements, elapsed_ms = run_vlm_scene_check(
            self.image_path,
            shot,
            vlm_scorer=None,
        )

        self.assertEqual(len(findings), 0)
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "vlm_scene")
        self.assertEqual(abstentions[0].reason, "vlm_backend_unavailable")
        self.assertFalse(measurements["vlm_backend_available"])

        # Integration through judge()
        # When vlm_scorer is None and environment has no Qwen3 GPU weights,
        # _resolve_vlm_scorer returns None or the stub backend is None:
        # Mocking _resolve_vlm_scorer by ensuring judge receives None when backend cannot run
        request = JuryRequest(
            schema_version="2.0",
            image_path=str(self.image_path),
            shot=shot,
            checks=("vlm_scene",),
            budget="full",
        )
        # Passing vlm_scorer=None when dependencies are not installed / model not downloaded
        # _resolve_vlm_scorer returns None or Qwen3 without weights.
        # Let's test the None case directly in judge by stubbing _resolve_vlm_scorer returning None
        from unittest.mock import patch

        with patch("frame_jury.jury._resolve_vlm_scorer", return_value=None):
            verdict = judge(request)

        self.assertEqual(verdict.verdict, "unsure")
        self.assertEqual(len(verdict.findings), 0)
        self.assertEqual(len(verdict.abstentions), 1)
        self.assertEqual(verdict.abstentions[0].check, "vlm_scene")
        self.assertEqual(verdict.abstentions[0].reason, "vlm_backend_unavailable")

    def test_cheap_budget_never_calls_scorer(self) -> None:
        """Under budget='cheap', the VLM scorer must never be called."""
        shot = _make_shot(characters=["Hero"])
        scorer = StubVlmScorer(scores=0.99)

        # 1. checks includes vlm_scene explicitly, but budget is cheap
        request = JuryRequest(
            schema_version="2.0",
            image_path=str(self.image_path),
            shot=shot,
            checks=("presence", "vlm_scene"),
            budget="cheap",
        )
        # Supply a detector that finds 1 person so presence does not block
        from frame_jury.backends.base import Detection, DetectorBackend

        class StubDet(DetectorBackend):
            def name(self) -> str:
                return "stub"

            def weights_sha256(self) -> str:
                return "stub-sha"

            def detect(self, image_path: str | Path, *, score_threshold: float = 0.5) -> list[Detection]:
                return [Detection("person", 0.9, (10, 10, 100, 100))]

        verdict = judge(request, detector=StubDet(), vlm_scorer=scorer)

        self.assertEqual(
            scorer.call_count,
            0,
            "VLM scorer was called under budget='cheap'!",
        )
        # No vlm_scene findings or abstentions
        self.assertFalse(any(f.check == "vlm_scene" for f in verdict.findings))
        self.assertFalse(any(a.check == "vlm_scene" for a in verdict.abstentions))

    def test_custom_calibration_thresholds(self) -> None:
        """Thresholds from a custom calibration file are respected."""
        shot = _make_shot(characters=["Hero"], framing="close-up")
        scorer = StubVlmScorer(scores={"Does any ONE of": 0.60})

        cal_dict: dict[str, Any] = {
            "fitted": False,
            "fitted_at": None,
            "label_file": None,
            "thresholds": {
                "_default": {
                    "person_score_threshold": 0.5,
                    "missing_entity_abstain_on_empty": True,
                    "extra_person_confidence": 0.8,
                    "duplicated_character_confidence": 0.85,
                    "missing_entity_confidence": 0.6,
                    "identity_similarity_threshold": 0.55,
                    "identity_ambiguous_band": 0.05,
                    "identity_confidence": 0.85,
                    "vlm_duplicated_character_threshold": 0.70,  # custom higher threshold
                    "vlm_missing_entity_threshold": 0.70,
                }
            },
        }
        import json

        cal_file = self.tmp / "custom_cal.json"
        cal_file.write_text(json.dumps(cal_dict), encoding="utf-8")
        custom_cal = CalibrationFile.load(cal_file)

        request = JuryRequest(
            schema_version="2.0",
            image_path=str(self.image_path),
            shot=shot,
            checks=("vlm_scene",),
            budget="full",
        )
        # Score is 0.60, custom threshold is 0.70 -> accept
        verdict = judge(request, vlm_scorer=scorer, calibration=custom_cal)
        self.assertEqual(verdict.verdict, "accept")
        self.assertEqual(len(verdict.findings), 0)


class TestVlmQwen3Backend(unittest.TestCase):
    """Unit tests for frame_jury.backends.vlm_qwen3 backend module."""

    def test_constants_and_lazy_import(self) -> None:
        """Module can be imported without downloading model weights or requiring torch."""
        self.assertEqual(MODEL_ID, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertRegex(MODEL_REVISION, r"^[0-9a-f]{40}$")

        scorer = Qwen3VlmScorer()
        self.assertEqual(scorer.name(), "qwen3-vl-8b-instruct")
        self.assertEqual(scorer.revision(), MODEL_REVISION)
        self.assertEqual(scorer.weights_sha256(), MODEL_REVISION)
        # Model and processor must remain uninitialized until score_yes_no is invoked
        self.assertIsNone(scorer._model)
        self.assertIsNone(scorer._processor)


if __name__ == "__main__":
    unittest.main()
