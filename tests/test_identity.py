"""Tests for M3 — identity check and bipartite optimal assignment.

All tests:
  - use no network and no GPU (AGENTS.md quality bar);
  - use synthetic fixtures and small dummy images constructed in the test;
  - use a fake FaceBackend returning fixed detections/embeddings so models
    are never downloaded;
  - are deterministic;
  - cover defect detection, all abstention paths, calibration differences,
    input immutability, bipartite assignment, and router integration.
"""

from __future__ import annotations

import copy
import json
import math
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Any, Callable

from frame_jury.backends.base import Detection, FaceBackend
from frame_jury.backends.face_yunet_sface import YuNetSFaceBackend
from frame_jury.calibration.thresholds import CalibrationFile, load_defaults
from frame_jury.checks.identity import run_identity_check
from frame_jury.contract import (
    DEFECT_EXTRA_PERSON,
    DEFECT_WRONG_IDENTITY,
    Abstention,
    Entity,
    Finding,
    JuryRequest,
    SCHEMA_VERSION,
    Shot,
    Staging,
    VerdictBuilder,
)
from frame_jury.jury import judge

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _png_1x1() -> bytes:
    """Produce a minimal valid 1x1 PNG."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    sig = b"\x89PNG\r\n\x1a\n"
    hdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes((0, 128, 128, 128)))
    return sig + chunk(b"IHDR", hdr) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def _make_dummy_image(tmp: Path, name: str) -> Path:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_png_1x1())
    return p


class FakeFaceBackend(FaceBackend):
    """Stub FaceBackend returning fixed detections and embeddings without network or GPU."""

    def __init__(
        self,
        *,
        frame_faces: list[Detection] | None = None,
        reference_faces: list[Detection] | None = None,
        frame_embedding: list[float] | None = None,
        reference_embedding: list[float] | None = None,
        name: str = "fake-face-backend",
        sha256: str = "0" * 64,
        ref_embed_map: dict[str, list[float]] | None = None,
        frame_embed_map: dict[tuple[int, int, int, int], list[float]] | None = None,
        frame_embeddings: dict[tuple[int, int, int, int], list[float]] | None = None,
        ref_embeddings: dict[str, list[float]] | None = None,
        similarity_fn: Callable[[list[float], list[float]], float] | None = None,
    ) -> None:
        self._frame_faces = (
            frame_faces
            if frame_faces is not None
            else [Detection(label="face", confidence=0.95, box=(10, 10, 50, 50))]
        )
        self._reference_faces = (
            reference_faces
            if reference_faces is not None
            else [Detection(label="face", confidence=0.98, box=(20, 20, 60, 60))]
        )
        # Default unit vectors:
        # ref = [1, 0, 0]
        # frame = [s, sqrt(1-s^2), 0] -> dot product = s
        self._ref_emb = reference_embedding if reference_embedding is not None else [1.0, 0.0, 0.0]
        self._frame_emb = frame_embedding if frame_embedding is not None else [0.8, 0.6, 0.0]
        self._backend_name = name
        self._sha256 = sha256
        self._ref_embed_map = ref_embed_map or ref_embeddings
        self._frame_embed_map = frame_embed_map or frame_embeddings
        self._similarity_fn = similarity_fn
        self.detect_calls: list[str] = []
        self.embed_calls: list[str] = []

    def name(self) -> str:
        return self._backend_name

    def weights_sha256(self) -> str:
        return self._sha256

    def detect_faces(self, image_path: str | Path) -> list[Detection]:
        p = str(image_path)
        self.detect_calls.append(p)
        if "ref" in p:
            return list(self._reference_faces)
        return list(self._frame_faces)

    def embed(
        self, image_path: str | Path, box: tuple[int, int, int, int]
    ) -> list[float]:
        p = str(image_path)
        self.embed_calls.append(p)
        if "ref" in p:
            if self._ref_embed_map:
                for k, v in self._ref_embed_map.items():
                    if k in p:
                        return list(v)
            return list(self._ref_emb)
        if self._frame_embed_map and box in self._frame_embed_map:
            return list(self._frame_embed_map[box])
        return list(self._frame_emb)

    def similarity(self, a: list[float], b: list[float]) -> float:
        if self._similarity_fn is not None:
            return self._similarity_fn(a, b)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    @classmethod
    def with_target_similarity(
        cls,
        target_similarity: float,
        *,
        frame_faces: list[Detection] | None = None,
        reference_faces: list[Detection] | None = None,
    ) -> "FakeFaceBackend":
        """Construct a stub producing exact target cosine similarity."""
        s = max(-1.0, min(1.0, target_similarity))
        rem = math.sqrt(max(0.0, 1.0 - s * s))
        return cls(
            reference_embedding=[1.0, 0.0, 0.0],
            frame_embedding=[s, rem, 0.0],
            frame_faces=frame_faces,
            reference_faces=reference_faces,
        )


def _make_shot(
    *,
    framing: str = "close-up",
    reference_images: tuple[str, ...] = ("ref.png",),
    entity_id: str = "char-vigia",
    display_name: str = "O vigia",
    other_people_allowed: bool = True,
) -> Shot:
    return Shot(
        shot_id="shot-001",
        framing=framing,
        declared_entities=(
            Entity(
                entity_id=entity_id,
                kind="character",
                display_name=display_name,
                aliases=(),
                reference_images=reference_images,
            ),
        ),
        staging=Staging(purpose="test", must_render=(), composition=()),
        positive_prompt="portrait",
        negative_prompt="",
        other_people_allowed=other_people_allowed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestIdentityCheck(unittest.TestCase):
    """Direct unit tests for run_identity_check."""

    def test_wrong_identity_produces_blocking_finding_and_evidence(self) -> None:
        """Similarity below threshold produces wrong_identity with numeric evidence."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            # Close-up threshold is 0.60, band 0.05 -> below 0.55 triggers wrong_identity.
            backend = FakeFaceBackend.with_target_similarity(0.35)

            findings, abstentions, measurements, elapsed_ms = run_identity_check(
                frame_path, shot, backend
            )

        self.assertEqual(len(findings), 1)
        self.assertEqual(len(abstentions), 0)
        f = findings[0]
        self.assertEqual(f.check, "identity")
        self.assertEqual(f.defect, DEFECT_WRONG_IDENTITY)
        self.assertEqual(f.severity, "blocking")
        self.assertEqual(f.entity_id, "char-vigia")
        self.assertIn("similarity", f.evidence)
        self.assertAlmostEqual(f.evidence["similarity"], 0.35, places=2)
        self.assertEqual(f.evidence["threshold"], 0.60)
        self.assertEqual(f.evidence["framing"], "close-up")
        self.assertEqual(f.evidence["reference_path"], str(ref_path))
        self.assertIn("boxes", f.evidence)
        self.assertTrue(len(f.evidence["boxes"]) > 0)
        self.assertTrue(len(f.prompt_hint) > 0)
        self.assertGreaterEqual(elapsed_ms, 0.0)
        self.assertAlmostEqual(measurements["identity_similarity"], 0.35, places=2)

    def test_accept_when_above_threshold(self) -> None:
        """Similarity above threshold + band produces no defect findings and no abstentions."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            # Close-up threshold is 0.60, band 0.05 -> 0.85 is safely above threshold.
            backend = FakeFaceBackend.with_target_similarity(0.85)

            findings, abstentions, measurements, elapsed_ms = run_identity_check(
                frame_path, shot, backend
            )

        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])
        self.assertAlmostEqual(measurements["identity_similarity"], 0.85, places=2)
        self.assertNotIn("identity_abstain_reason", measurements)

    def test_no_face_in_frame_abstains(self) -> None:
        """No face in frame emits first-class Abstention and reason in measurements."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            backend = FakeFaceBackend(frame_faces=[])

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "identity")
        self.assertEqual(abstentions[0].reason, "no_face_in_frame")
        self.assertEqual(abstentions[0].entity_id, "char-vigia")
        self.assertTrue(measurements.get("identity_abstain"))
        self.assertEqual(measurements.get("identity_abstain_reason"), "no_face_in_frame")

    def test_no_face_in_reference_image_abstains(self) -> None:
        """No face in reference image emits first-class Abstention and reason."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            backend = FakeFaceBackend(reference_faces=[])

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "identity")
        self.assertEqual(abstentions[0].reason, "no_face_in_reference")
        self.assertEqual(abstentions[0].entity_id, "char-vigia")
        self.assertTrue(measurements.get("identity_abstain"))
        self.assertEqual(measurements.get("identity_abstain_reason"), "no_face_in_reference")

    def test_no_reference_image_declared_abstains(self) -> None:
        """Entity without reference_images emits first-class Abstention."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")

            shot = _make_shot(framing="close-up", reference_images=())
            backend = FakeFaceBackend()

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "identity")
        self.assertEqual(abstentions[0].reason, "no_reference_image")
        self.assertEqual(abstentions[0].entity_id, "char-vigia")
        self.assertTrue(measurements.get("identity_abstain"))
        self.assertEqual(measurements.get("identity_abstain_reason"), "no_reference_image")

    def test_ambiguous_band_around_threshold_abstains(self) -> None:
        """Similarity inside ambiguous band emits first-class Abstention rather than guessing."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            # Close-up threshold 0.60, band 0.05 -> [0.55, 0.65] is ambiguous band.
            backend = FakeFaceBackend.with_target_similarity(0.61)

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "identity")
        self.assertEqual(abstentions[0].reason, "ambiguous_band")
        self.assertEqual(abstentions[0].entity_id, "char-vigia")
        self.assertTrue(measurements.get("identity_abstain"))
        self.assertEqual(measurements.get("identity_abstain_reason"), "ambiguous_band")

    def test_wide_shot_differs_from_close_up_threshold(self) -> None:
        """Wide shot threshold differs from close-up and is more permissive."""
        cal = load_defaults()
        close_up_th = cal.for_framing("close-up").identity_similarity_threshold
        wide_shot_th = cal.for_framing("wide shot").identity_similarity_threshold

        self.assertNotEqual(close_up_th, wide_shot_th)
        self.assertLess(wide_shot_th, close_up_th)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            # Intermediate similarity (0.52):
            # close-up: th=0.60, band=0.05 -> lower_band=0.55 -> 0.52 is below lower_band -> wrong_identity!
            # wide shot: th=0.45, band=0.05 -> upper_band=0.50 -> 0.52 is above upper_band -> clean accept!
            backend = FakeFaceBackend.with_target_similarity(0.52)

            shot_close = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            findings_close, abstentions_close, _, _ = run_identity_check(frame_path, shot_close, backend, calibration=cal)
            self.assertEqual(len(findings_close), 1)
            self.assertEqual(len(abstentions_close), 0)
            self.assertEqual(findings_close[0].severity, "blocking")

            shot_wide = _make_shot(framing="wide shot", reference_images=(str(ref_path),))
            findings_wide, abstentions_wide, _, _ = run_identity_check(frame_path, shot_wide, backend, calibration=cal)
            self.assertEqual(len(findings_wide), 0)
            self.assertEqual(len(abstentions_wide), 0)

    def test_check_never_mutates_inputs(self) -> None:
        """run_identity_check must never modify shot, entities, or paths."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_path = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref_path),))
            backend = FakeFaceBackend.with_target_similarity(0.70)
            cal = load_defaults()

            shot_clone = copy.deepcopy(shot)
            cal_clone = copy.deepcopy(cal)

            run_identity_check(frame_path, shot, backend, calibration=cal)

            self.assertEqual(shot, shot_clone)
            self.assertEqual(cal, cal_clone)

    def test_shot_with_no_characters_clean(self) -> None:
        """Shots without characters return no findings, no abstentions, and run cleanly."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")

            shot = Shot(
                shot_id="shot-env",
                framing="wide shot",
                declared_entities=(
                    Entity(
                        entity_id="obj-notebook",
                        kind="object",
                        display_name="Notebook",
                        aliases=(),
                        reference_images=(),
                    ),
                ),
                staging=Staging(purpose="prop shot", must_render=(), composition=()),
                positive_prompt="notebook on table",
                negative_prompt="",
            )
            backend = FakeFaceBackend()
            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)
            self.assertEqual(findings, [])
            self.assertEqual(abstentions, [])
            self.assertEqual(measurements.get("declared_characters"), 0)


class TestBipartiteIdentityAssignment(unittest.TestCase):
    """Tests for Decision 1 (optimal bipartite assignment) and Decision 2 (first-class abstention)."""

    def _make_two_character_shot(
        self,
        ref_a: Path,
        ref_b: Path,
        *,
        framing: str = "close-up",
    ) -> Shot:
        return Shot(
            shot_id="shot-duo",
            framing=framing,
            declared_entities=(
                Entity(
                    entity_id="char-alice",
                    kind="character",
                    display_name="Alice",
                    aliases=(),
                    reference_images=(str(ref_a),),
                ),
                Entity(
                    entity_id="char-bob",
                    kind="character",
                    display_name="Bob",
                    aliases=(),
                    reference_images=(str(ref_b),),
                ),
            ),
            staging=Staging(purpose="two characters", must_render=(), composition=()),
            positive_prompt="Alice and Bob conversing",
            negative_prompt="",
        )

    def test_two_declared_characters_and_two_faces_never_both_assigned_same_face(self) -> None:
        """Two declared characters competing for the same best face are assigned distinct faces."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_a = _make_dummy_image(root, "ref_alice.png")
            ref_b = _make_dummy_image(root, "ref_bob.png")

            shot = self._make_two_character_shot(ref_a, ref_b)

            face_0 = Detection(label="face", confidence=0.95, box=(0, 0, 50, 50))
            face_1 = Detection(label="face", confidence=0.95, box=(60, 60, 110, 110))

            # Both Alice and Bob have highest similarity to Face 0:
            # Alice: Face 0 = 0.90, Face 1 = 0.80
            # Bob:   Face 0 = 0.85, Face 1 = 0.10
            # Under greedy, both would grab Face 0.
            # Under Hungarian, Alice -> Face 1 (0.80), Bob -> Face 0 (0.85).
            def sim_fn(frame_emb: list[float], ref_emb: list[float]) -> float:
                face_id = int(frame_emb[0])
                char_id = int(ref_emb[0])
                table = {
                    (0, 0): 0.90,  # Face 0, Alice
                    (1, 0): 0.80,  # Face 1, Alice
                    (0, 1): 0.85,  # Face 0, Bob
                    (1, 1): 0.10,  # Face 1, Bob
                }
                return table.get((face_id, char_id), 0.0)

            backend = FakeFaceBackend(
                frame_faces=[face_0, face_1],
                frame_embeddings={face_0.box: [0.0], face_1.box: [1.0]},
                ref_embed_map={str(ref_a): [0.0], str(ref_b): [1.0]},
                similarity_fn=sim_fn,
            )

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        # Both characters are assigned distinct faces and both pass threshold (0.80 > 0.65 and 0.85 > 0.65).
        # Neither produces a false wrong_identity defect!
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])

    def test_rectangular_more_characters_than_faces_surplus_abstains(self) -> None:
        """Rectangular case: more characters than faces leaves the surplus character abstaining."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_a = _make_dummy_image(root, "ref_alice.png")
            ref_b = _make_dummy_image(root, "ref_bob.png")

            shot = self._make_two_character_shot(ref_a, ref_b)

            # Only ONE face detected in frame:
            only_face = Detection(label="face", confidence=0.95, box=(10, 10, 60, 60))

            # Alice matches this face well (0.85); Bob matches poorly (0.20)
            def sim_fn(frame_emb: list[float], ref_emb: list[float]) -> float:
                char_id = int(ref_emb[0])
                return 0.85 if char_id == 0 else 0.20

            backend = FakeFaceBackend(
                frame_faces=[only_face],
                frame_embeddings={only_face.box: [0.0]},
                ref_embed_map={str(ref_a): [0.0], str(ref_b): [1.0]},
                similarity_fn=sim_fn,
            )

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        # Alice was assigned the face and matched cleanly.
        # Bob was left unassigned: surplus abstains (reason: "unassigned_face"), NOT wrong_identity!
        self.assertEqual(findings, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0].check, "identity")
        self.assertEqual(abstentions[0].reason, "unassigned_face")
        self.assertEqual(abstentions[0].entity_id, "char-bob")

    def test_assignment_picks_higher_total_over_greedy(self) -> None:
        """The assignment picks the higher-total pairing over greedy when they differ."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame_path = _make_dummy_image(root, "frame.png")
            ref_a = _make_dummy_image(root, "ref_alice.png")
            ref_b = _make_dummy_image(root, "ref_bob.png")

            shot = self._make_two_character_shot(ref_a, ref_b)

            face_0 = Detection(label="face", confidence=0.95, box=(0, 0, 50, 50))
            face_1 = Detection(label="face", confidence=0.95, box=(60, 60, 110, 110))

            # Alice: Face 0 = 0.90, Face 1 = 0.80
            # Bob:   Face 0 = 0.85, Face 1 = 0.10
            # Greedy: Alice->Face 0 (0.90), Bob->Face 1 (0.10) => sum = 1.00 (Bob fails threshold 0.60!)
            # Optimal: Alice->Face 1 (0.80), Bob->Face 0 (0.85) => sum = 1.65 (both pass threshold!)
            def sim_fn(frame_emb: list[float], ref_emb: list[float]) -> float:
                face_id = int(frame_emb[0])
                char_id = int(ref_emb[0])
                table = {
                    (0, 0): 0.90,
                    (1, 0): 0.80,
                    (0, 1): 0.85,
                    (1, 1): 0.10,
                }
                return table.get((face_id, char_id), 0.0)

            backend = FakeFaceBackend(
                frame_faces=[face_0, face_1],
                frame_embeddings={face_0.box: [0.0], face_1.box: [1.0]},
                ref_embed_map={str(ref_a): [0.0], str(ref_b): [1.0]},
                similarity_fn=sim_fn,
            )

            findings, abstentions, measurements, _ = run_identity_check(frame_path, shot, backend)

        # Because optimal bipartite assignment found the higher total (1.65), both pass cleanly.
        self.assertEqual(findings, [])
        self.assertEqual(abstentions, [])

    def test_verdict_with_only_abstentions_is_unsure(self) -> None:
        """A verdict containing only abstentions resolves to unsure (SPEC.md §4)."""
        b = VerdictBuilder("shot-test-abstain")
        b.add_abstention(
            Abstention(
                check="identity",
                reason="no_face_in_frame",
                entity_id="char-vigia",
                explanation="no face found in frame",
            )
        )
        verdict = b.build()
        self.assertEqual(verdict.verdict, "unsure")
        self.assertEqual(len(verdict.findings), 0)
        self.assertEqual(len(verdict.abstentions), 1)

    def test_verdict_with_warning_and_no_abstention_is_accept(self) -> None:
        """A verdict with a warning finding and no abstention is accept (SPEC.md §4)."""
        b = VerdictBuilder("shot-test-warning")
        b.add_finding(
            Finding(
                check="presence",
                defect="non_blocking_warning",
                severity="warning",
                confidence=0.75,
                evidence={},
                explanation="a non-blocking defect occurred",
            )
        )
        verdict = b.build()
        self.assertEqual(verdict.verdict, "accept")
        self.assertEqual(len(verdict.findings), 1)
        self.assertEqual(verdict.findings[0].severity, "warning")
        self.assertEqual(len(verdict.abstentions), 0)


class TestJuryIdentityIntegration(unittest.TestCase):
    """Tests routing and integration through frame_jury.jury.judge."""

    def _build_request(self, frame_path: Path, shot: Shot, checks: list[str]) -> JuryRequest:
        return JuryRequest.from_dict({
            "schema_version": SCHEMA_VERSION,
            "image_path": str(frame_path),
            "shot": shot.to_dict(),
            "checks": checks,
            "budget": "cheap",
        })

    def test_judge_rejects_on_wrong_identity(self) -> None:
        """Verdict is reject when identity similarity is below threshold."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame = _make_dummy_image(root, "frame.png")
            ref = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref),))
            request = self._build_request(frame, shot, ["identity"])
            backend = FakeFaceBackend.with_target_similarity(0.30)

            verdict = judge(request, face_backend=backend)

        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.findings), 1)
        self.assertEqual(verdict.findings[0].defect, DEFECT_WRONG_IDENTITY)
        self.assertEqual(len(verdict.abstentions), 0)
        self.assertIn("identity", verdict.timings_ms)
        self.assertTrue(any(det["check"] == "identity" for det in verdict.detectors))

    def test_judge_unsure_on_identity_abstain(self) -> None:
        """Verdict is unsure when identity check abstains via Abstention."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame = _make_dummy_image(root, "frame.png")
            ref = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref),))
            request = self._build_request(frame, shot, ["identity"])
            backend = FakeFaceBackend(frame_faces=[])

            verdict = judge(request, face_backend=backend)

        self.assertEqual(verdict.verdict, "unsure")
        self.assertEqual(len(verdict.findings), 0)
        self.assertEqual(len(verdict.abstentions), 1)
        self.assertEqual(verdict.abstentions[0].check, "identity")
        self.assertEqual(verdict.abstentions[0].reason, "no_face_in_frame")

    def test_cheap_budget_early_stop_presence_before_identity(self) -> None:
        """Cheap budget stops after blocking presence defect; identity is not run."""
        from frame_jury.backends.base import DetectorBackend

        class StubPresenceDetector(DetectorBackend):
            def name(self) -> str:
                return "stub-presence"

            def weights_sha256(self) -> str:
                return "0" * 64

            def detect(self, image_path, *, score_threshold=0.5):
                # 2 people detected for 1 declared character with other_people_allowed=False -> extra_person (blocking)
                return [
                    Detection(label="person", confidence=0.9, box=(0, 0, 50, 50)),
                    Detection(label="person", confidence=0.9, box=(60, 0, 100, 50)),
                ]

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame = _make_dummy_image(root, "frame.png")
            ref = _make_dummy_image(root, "ref.png")

            shot = _make_shot(framing="close-up", reference_images=(str(ref),), other_people_allowed=False)
            request = self._build_request(frame, shot, ["presence", "identity"])
            face_backend = FakeFaceBackend.with_target_similarity(0.20)

            verdict = judge(
                request,
                detector=StubPresenceDetector(),
                face_backend=face_backend,
            )

        # Presence failed (blocking), so router stopped early.
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(len(verdict.findings), 1)
        self.assertEqual(verdict.findings[0].defect, DEFECT_EXTRA_PERSON)
        # Identity did not run:
        self.assertNotIn("identity", verdict.timings_ms)
        self.assertEqual(len(face_backend.detect_calls), 0)


class TestFaceBackendCosineMath(unittest.TestCase):
    """Test cosine similarity math on YuNetSFaceBackend."""

    def test_similarity_math(self) -> None:
        backend = YuNetSFaceBackend()

        # Identical
        self.assertAlmostEqual(backend.similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)
        # Orthogonal
        self.assertAlmostEqual(backend.similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        # Opposite
        self.assertAlmostEqual(backend.similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        # Zero norm
        self.assertAlmostEqual(backend.similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestYuNetSFaceBackend(unittest.TestCase):
    """Unit tests for YuNetSFaceBackend methods and helpers."""

    def test_metadata(self) -> None:
        backend = YuNetSFaceBackend()
        self.assertEqual(backend.name(), "yunet-sface")
        self.assertTrue(backend.weights_sha256().startswith("yunet:"))
        self.assertIn("sface:", backend.weights_sha256())

    def test_sha256_file(self) -> None:
        import hashlib
        from frame_jury.backends.face_yunet_sface import _sha256_file
        content = b"frame-jury-m3"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            p = Path(f.name)
        try:
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(_sha256_file(p), expected)
        finally:
            p.unlink(missing_ok=True)

    def test_ensure_weights_sha256_mismatch_raises(self) -> None:
        from frame_jury.backends.face_yunet_sface import _ensure_weights, _YUNET_FILENAME
        with tempfile.TemporaryDirectory() as d:
            cache_dir = Path(d)
            # Create a file with bad content
            bad_file = cache_dir / _YUNET_FILENAME
            bad_file.write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError) as ctx:
                _ensure_weights(cache_dir)
            self.assertIn("mismatch", str(ctx.exception))


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
