"""tests.test_cli — the batch command line (`python -m frame_jury judge`).

All tests monkeypatch ``frame_jury.jury.judge`` and the three private
resolvers on ``frame_jury.jury``, so no torch, OpenCV, model weights or
network are ever loaded (AGENTS.md quality bar).  The patched resolvers
stand in for the heavy backends; the patched ``judge`` receives whatever the
CLI resolved and returns a small real :class:`Verdict`.

Covered:
  two valid requests       — verdicts in request order, exit 0, every
                             resolver built at most once, shared instances,
                             one stderr progress line per frame;
  a failing ``judge`` call — that entry carries ``error``, the others carry
                             ``verdict``, exit 2;
  duplicate request_id     — exit 1, nothing judged, output not written;
  no ``budget: "full"``    — the VLM resolver is never called.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from frame_jury.__main__ import main
from frame_jury.contract import Verdict


def _request(
    request_id: str,
    *,
    checks: list[str],
    budget: str = "cheap",
) -> dict[str, Any]:
    """One minimal section 4 request, plus the caller-chosen ``request_id``."""
    return {
        "request_id": request_id,
        "schema_version": "2.0",
        "image_path": "frame.png",
        "shot": {
            "shot_id": request_id,
            "framing": "close-up",
            "declared_entities": [
                {
                    "entity_id": "char-keeper",
                    "kind": "character",
                    "display_name": "The keeper",
                    "aliases": [],
                    "reference_images": [],
                }
            ],
            "staging": {"purpose": "Test purpose.", "must_render": [], "composition": []},
            "positive_prompt": "test",
            "negative_prompt": "",
        },
        "checks": checks,
        "budget": budget,
    }


def _verdict(shot_id: str, outcome: str = "accept") -> Verdict:
    """A minimal real verdict, as the real ``judge`` would build."""
    return Verdict(
        schema_version="2.0",
        shot_id=shot_id,
        verdict=outcome,  # type: ignore[arg-type]
        confidence=0.99,
    )


class TestJudgeCommand(unittest.TestCase):
    """``python -m frame_jury judge`` against patched resolvers and judge."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.requests_path = self.tmp / "requests.json"
        self.out_path = self.tmp / "verdicts.json"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_requests(self, entries: list[dict[str, Any]]) -> None:
        self.requests_path.write_text(json.dumps(entries), encoding="utf-8")

    def _run(self, stderr: io.StringIO | None = None) -> int:
        """Call main() the way the process would; optionally capture stderr."""
        argv = ["judge", "--requests", str(self.requests_path), "--out", str(self.out_path)]
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr or io.StringIO()):
            return main(argv)

    def _patch_jobs(
        self, judge_side_effect: Any
    ) -> tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock, mock.MagicMock]:
        """Patch judge and the three resolvers; return (judge, detector, face, vlm) mocks."""
        judge_patch = mock.patch("frame_jury.jury.judge", side_effect=judge_side_effect)
        detector_patch = mock.patch(
            "frame_jury.jury._resolve_detector",
            return_value=mock.MagicMock(name="detector"),
        )
        face_patch = mock.patch(
            "frame_jury.jury._resolve_face_backend",
            return_value=mock.MagicMock(name="face_backend"),
        )
        vlm_patch = mock.patch(
            "frame_jury.jury._resolve_vlm_scorer",
            return_value=mock.MagicMock(name="vlm_scorer"),
        )
        return judge_patch, detector_patch, face_patch, vlm_patch

    def test_two_requests_produce_two_verdicts_and_backends_load_once(self) -> None:
        """Two valid requests: entries in order, exit 0, each resolver used at most once."""
        self._write_requests(
            [
                _request("req-a", checks=["presence"]),
                _request("req-b", checks=["identity"]),
            ]
        )
        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(
            lambda request, **_: _verdict(request.shot.shot_id)
        )
        stderr = io.StringIO()

        with judge_patch as judge, detector_patch as resolve_detector, \
                face_patch as resolve_face, vlm_patch as resolve_vlm:
            exit_code = self._run(stderr)

        self.assertEqual(exit_code, 0)

        entries = json.loads(self.out_path.read_text(encoding="utf-8"))
        self.assertEqual([entry["request_id"] for entry in entries], ["req-a", "req-b"])
        self.assertNotIn("error", entries[0])
        self.assertNotIn("error", entries[1])
        self.assertEqual(entries[0]["verdict"]["verdict"], "accept")
        self.assertEqual(entries[0]["verdict"]["shot_id"], "req-a")
        self.assertEqual(entries[1]["verdict"]["shot_id"], "req-b")

        # The backends load once for the batch, not once per frame, and only
        # when a request needs them: the first request runs presence, the
        # second identity, and nobody asks for the VLM.
        resolve_detector.assert_called_once_with(None)
        resolve_face.assert_called_once_with(None)
        resolve_vlm.assert_not_called()
        self.assertEqual(judge.call_count, 2)

        # Both judge calls share one detector and one face backend instance.
        detectors_passed = [
            call.kwargs["detector"]
            for call in judge.call_args_list
            if call.kwargs["detector"] is not None
        ]
        self.assertEqual(detectors_passed, [resolve_detector.return_value])
        faces_passed = [
            call.kwargs["face_backend"]
            for call in judge.call_args_list
            if call.kwargs["face_backend"] is not None
        ]
        self.assertEqual(faces_passed, [resolve_face.return_value])

        # The temporary file is replaced away after the final write.
        self.assertFalse(self.out_path.with_name(self.out_path.name + ".tmp").exists())

        # One stderr progress line per frame, in the specified format.
        self.assertEqual(len(stderr.getvalue().splitlines()), 2)
        self.assertRegex(
            stderr.getvalue(),
            r"^judged 1/2 req-a: accept \(0 findings, 0 abstentions\) in \d+\.\ds\n"
            r"judged 2/2 req-b: accept \(0 findings, 0 abstentions\) in \d+\.\ds$",
        )

    def test_failing_request_reports_error_and_batch_continues(self) -> None:
        """A request whose judge raises: error entry, others judged, exit 2."""
        self._write_requests(
            [
                _request("bad", checks=["presence"]),
                _request("good", checks=["presence"]),
            ]
        )

        def fake_judge(request: Any, **_: Any) -> Verdict:
            if request.shot.shot_id == "bad":
                raise RuntimeError("weights file missing")
            return _verdict(request.shot.shot_id)

        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(fake_judge)

        with judge_patch, detector_patch, face_patch, vlm_patch:
            exit_code = self._run()

        self.assertEqual(exit_code, 2)
        entries = json.loads(self.out_path.read_text(encoding="utf-8"))
        self.assertEqual([entry["request_id"] for entry in entries], ["bad", "good"])
        self.assertEqual(entries[0]["error"], "RuntimeError: weights file missing")
        self.assertNotIn("error", entries[1])
        self.assertEqual(entries[1]["verdict"]["verdict"], "accept")

    def test_duplicate_request_id_exits_1_without_judging_or_writing(self) -> None:
        """Duplicate request_id: exit 1, nothing judged, output file not written."""
        self._write_requests(
            [
                _request("same", checks=["presence"]),
                _request("same", checks=["presence"]),
            ]
        )
        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(_verdict)

        with judge_patch as judge, detector_patch, face_patch, vlm_patch:
            exit_code = self._run()

        self.assertEqual(exit_code, 1)
        self.assertFalse(self.out_path.exists())
        judge.assert_not_called()

    def test_missing_request_id_exits_1_without_judging(self) -> None:
        """A request without request_id is a usage error like a duplicate one."""
        entry = _request("has-id", checks=["presence"])
        del entry["request_id"]
        self._write_requests([entry])
        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(_verdict)

        with judge_patch as judge, detector_patch, face_patch, vlm_patch:
            exit_code = self._run()

        self.assertEqual(exit_code, 1)
        self.assertFalse(self.out_path.exists())
        judge.assert_not_called()

    def test_invalid_json_requests_file_exits_1(self) -> None:
        """An unreadable or malformed request file is a usage error (exit 1)."""
        self.requests_path.write_text("{not json", encoding="utf-8")
        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(_verdict)

        with judge_patch as judge, detector_patch, face_patch, vlm_patch:
            exit_code = self._run()

        self.assertEqual(exit_code, 1)
        self.assertFalse(self.out_path.exists())
        judge.assert_not_called()

    def test_no_full_budget_means_vlm_resolver_never_called(self) -> None:
        """With no request at budget=full, the VLM resolver is never called."""
        self._write_requests(
            [
                _request("req-presence", checks=["presence"]),
                _request("req-cheap-scene", checks=["vlm_scene"]),  # cheap: VLM never needed
            ]
        )
        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(
            lambda request, **_: _verdict(request.shot.shot_id)
        )

        with judge_patch as judge, detector_patch, face_patch, vlm_patch as resolve_vlm:
            exit_code = self._run()

        self.assertEqual(exit_code, 0)
        resolve_vlm.assert_not_called()
        self.assertEqual(judge.call_count, 2)

    def test_full_budget_builds_the_vlm_backend_once(self) -> None:
        """A request at budget=full builds the VLM backend once and shares it."""
        self._write_requests(
            [
                _request("req-cheap", checks=["presence"]),
                _request("req-full", checks=["vlm_scene"], budget="full"),
            ]
        )
        judge_patch, detector_patch, face_patch, vlm_patch = self._patch_jobs(
            lambda request, **_: _verdict(request.shot.shot_id)
        )

        with judge_patch as judge, detector_patch as resolve_detector, \
                face_patch, vlm_patch as resolve_vlm:
            exit_code = self._run()

        self.assertEqual(exit_code, 0)
        resolve_vlm.assert_called_once_with(None)
        resolve_detector.assert_called_once_with(None)
        vlm_passed = [
            call.kwargs["vlm_scorer"]
            for call in judge.call_args_list
            if call.kwargs["vlm_scorer"] is not None
        ]
        self.assertEqual(vlm_passed, [resolve_vlm.return_value])


if __name__ == "__main__":
    unittest.main()
