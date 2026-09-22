from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from benchmarks.corpus.schema import write_cases
from benchmarks.label.faces import crop, face_boxes
from benchmarks.label.identity import (
    IDENTITY_SCHEMA_VERSION,
    IdentityLabelError,
    IdentityStore,
    identity_pairs,
    validate_identity_label,
)
from benchmarks.label.schema import LABEL_SCHEMA_VERSION, TAXONOMY_VERSION
from benchmarks.label.server import make_handler
from benchmarks.label.store import LabelStore
from frame_jury.backends.base import Detection, FaceBackend
from tests.helpers import make_case


class StubFaceBackend(FaceBackend):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def name(self) -> str:
        return "stub-face"

    def weights_sha256(self) -> str:
        return "0" * 64

    def detect_faces(self, image_path: str | Path) -> list[Detection]:
        self.calls.append(str(image_path))
        return [
            Detection("face", 0.9, (40, 10, 60, 30)),
            Detection("face", 0.8, (5, 8, 25, 28)),
        ]

    def embed(self, image_path: str | Path, box: tuple[int, int, int, int]) -> list[float]:
        return [1.0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 1.0


def make_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "navy").save(path)


def identity_case(root: Path, run_id: str, shot_id: str) -> dict[str, object]:
    case = make_case(root, run_id, shot_id)
    frame = Path(case["image_path"])
    reference = (root / run_id / f"{shot_id}-reference.png").resolve()
    make_image(frame)
    make_image(reference, (50, 70))
    entities = [
        {
            "entity_id": "char-main",
            "kind": "character",
            "display_name": "Main character",
            "aliases": [],
            "reference_images": [str(reference)],
        }
    ]
    case["shot"]["declared_entities"] = entities
    case["reference_images"] = {"char-main": [str(reference)]}
    return case


def identity_label(
    case_id: str = "run-a/shot-1",
    *,
    decision: str = "same",
    faces: list[int] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "case_id": case_id,
        "entity_id": "char-main",
        "decision": decision,
        "faces": [0] if faces is None else faces,
        "face_boxes": [[1, 2, 10, 12]],
        "labeller": "tester",
        "at": "2026-09-21T12:00:00Z",
    }


def frame_label(case_id: str) -> dict[str, object]:
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "case_id": case_id,
        "defects": ["clean"],
        "notes": "",
        "labeller": "tester",
        "at": "2026-09-21T12:00:00Z",
    }


class IdentityValidationTests(unittest.TestCase):
    def test_accepts_each_decision_with_its_allowed_faces(self) -> None:
        for decision, faces in (
            ("same", [0]),
            ("different", []),
            ("different", [0]),
            ("not_visible", []),
            ("unsure", []),
        ):
            with self.subTest(decision=decision, faces=faces):
                self.assertEqual(
                    validate_identity_label(
                        identity_label(decision=decision, faces=faces)
                    )["decision"],
                    decision,
                )

    def test_rejects_schema_decision_and_strict_field_errors(self) -> None:
        for field, value in (("schema_version", "2.0"), ("decision", "maybe")):
            record = identity_label()
            record[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(IdentityLabelError, field):
                validate_identity_label(record)

        missing = identity_label()
        missing.pop("entity_id")
        with self.assertRaisesRegex(IdentityLabelError, "entity_id"):
            validate_identity_label(missing)
        extra = identity_label()
        extra["notes"] = "unknown"
        with self.assertRaisesRegex(IdentityLabelError, "notes"):
            validate_identity_label(extra)

    def test_rejects_bad_face_boxes_indices_and_decision_cardinality(self) -> None:
        bad_box = identity_label()
        bad_box["face_boxes"] = [[1, 2, 3]]
        with self.assertRaisesRegex(IdentityLabelError, "face_boxes"):
            validate_identity_label(bad_box)

        for faces, message in (([1], "outside"), ([0, 0], "duplicate")):
            with self.subTest(faces=faces), self.assertRaisesRegex(IdentityLabelError, message):
                validate_identity_label(identity_label(faces=faces))

        with self.assertRaisesRegex(IdentityLabelError, "faces"):
            validate_identity_label(identity_label(decision="same", faces=[]))
        for decision in ("not_visible", "unsure"):
            with self.subTest(decision=decision), self.assertRaisesRegex(
                IdentityLabelError, "faces"
            ):
                validate_identity_label(identity_label(decision=decision, faces=[0]))

    def test_rejects_an_unknown_pair_only_when_known_pairs_are_supplied(self) -> None:
        record = identity_label()
        self.assertEqual(validate_identity_label(record)["case_id"], record["case_id"])
        with self.assertRaisesRegex(IdentityLabelError, "case_id/entity_id"):
            validate_identity_label(record, known_pairs={("other", "char-main")})


class IdentityPairAndStoreTests(unittest.TestCase):
    def test_identity_pairs_only_returns_reference_backed_characters_in_order(self) -> None:
        case = make_case(Path.cwd(), "run-a", "shot-1")
        case["shot"]["declared_entities"] = [
            {"entity_id": "obj", "kind": "object"},
            {"entity_id": "char-no-ref", "kind": "character"},
            {"entity_id": "char-b", "kind": "character"},
            {"entity_id": "char-a", "kind": "character"},
        ]
        case["reference_images"] = {
            "obj": ["object.png"],
            "char-no-ref": [],
            "char-b": ["b.png"],
            "char-a": ["a.png"],
        }
        self.assertEqual(identity_pairs(case), ["char-b", "char-a"])

    def test_store_prioritizes_frame_labelled_cases_and_skips_labelled_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            labels_path = root / "identity.jsonl"
            frame_labels_path = root / "labels.jsonl"
            first = identity_case(root, "run-a", "shot-1")
            prioritized = identity_case(root, "run-z", "shot-1")
            write_cases(cases_path, [first, prioritized])
            frame_labels_path.write_text(
                json.dumps(frame_label(prioritized["case_id"])) + "\n", encoding="utf-8"
            )

            store = IdentityStore(cases_path, labels_path, frame_labels_path)
            case, entity_id = store.next_pair()
            self.assertEqual(case["case_id"], prioritized["case_id"])
            self.assertEqual(entity_id, "char-main")
            store.append(identity_label(prioritized["case_id"]))
            self.assertEqual(store.next_pair()[0]["case_id"], first["case_id"])

    def test_store_counts_latest_pair_once_and_never_rewrites_existing_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            labels_path = root / "nested" / "identity.jsonl"
            case = identity_case(root, "run-a", "shot-1")
            write_cases(cases_path, [case])
            store = IdentityStore(cases_path, labels_path)
            store.append(identity_label(case["case_id"]))
            first_line = labels_path.read_text(encoding="utf-8").splitlines()[0]
            store.append(identity_label(case["case_id"], decision="different", faces=[]))
            lines = labels_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], first_line)
            self.assertEqual(len(lines), 2)
            self.assertEqual(store.labelled_count, 1)
            self.assertEqual(store.total_count, 1)


class FaceHelperTests(unittest.TestCase):
    def test_face_boxes_sorts_left_to_right_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "unique-cache-image.png"
            make_image(image)
            backend = StubFaceBackend()
            expected = [[5, 8, 25, 28], [40, 10, 60, 30]]
            self.assertEqual(face_boxes(str(image), backend), expected)
            self.assertEqual(face_boxes(str(image), backend), expected)
            self.assertEqual(backend.calls, [str(image)])

    def test_crop_returns_png_with_requested_longer_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            make_image(image, (100, 80))
            body = crop(str(image), [20, 20, 60, 40], pad=0.0, size=32)
            self.assertTrue(body.startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(BytesIO(body)) as result:
                self.assertEqual(max(result.size), 32)
                self.assertEqual(result.size, (32, 16))


class IdentityServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cases_path = self.root / "cases.jsonl"
        self.frame_labels_path = self.root / "labels.jsonl"
        self.identity_labels_path = self.root / "identity.jsonl"
        self.case = identity_case(self.root, "run-a", "shot-1")
        write_cases(self.cases_path, [self.case])
        self.store = LabelStore(self.cases_path, self.frame_labels_path)
        self.identity_store = IdentityStore(
            self.cases_path, self.identity_labels_path, self.frame_labels_path
        )
        self.backend = StubFaceBackend()

        from http.server import ThreadingHTTPServer

        handler = make_handler(
            self.store,
            "tester",
            identity_store=self.identity_store,
            face_backend_factory=lambda: self.backend,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def get_json(self, path: str) -> dict[str, object]:
        with urlopen(self.base_url + path) as response:
            return json.load(response)

    def get_text(self, path: str) -> str:
        with urlopen(self.base_url + path) as response:
            return response.read().decode("utf-8")

    def post_json(self, path: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def test_identity_api_returns_faces_and_appends_a_valid_record(self) -> None:
        response = self.get_json("/api/identity/next")
        self.assertEqual(response["case"]["case_id"], self.case["case_id"])
        self.assertEqual(response["entity_id"], "char-main")
        self.assertEqual(response["entity"]["reference_urls"], [
            f"/asset?case_id=run-a%2Fshot-1&kind=reference&entity_id=char-main&index=0"
        ])
        self.assertEqual([face["index"] for face in response["frame_faces"]], [0, 1])
        self.assertEqual(response["frame_faces"][0]["box"], [5, 8, 25, 28])
        self.assertEqual(response["reference_faces"][0]["index"], 0)
        self.assertEqual(response["labelled"], 0)
        self.assertEqual(response["total"], 1)
        self.assertEqual(response["labeller"], "tester")

        with urlopen(self.base_url + response["frame_faces"][0]["url"]) as asset:
            self.assertEqual(asset.headers.get_content_type(), "image/png")
            self.assertTrue(asset.read().startswith(b"\x89PNG\r\n\x1a\n"))
        with self.assertRaises(HTTPError) as bad_asset:
            urlopen(
                self.base_url
                + f"/asset?case_id=run-a%2Fshot-1&kind=frame_face&index=99"
            )
        self.assertEqual(bad_asset.exception.code, 404)

        status, result = self.post_json(
            "/api/identity/labels",
            {
                "case_id": self.case["case_id"],
                "entity_id": "char-main",
                "decision": "same",
                "faces": [0],
            },
        )
        self.assertEqual((status, result), (201, {"ok": True}))
        stored = json.loads(self.identity_labels_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["face_boxes"], [[5, 8, 25, 28], [40, 10, 60, 30]])
        self.assertEqual(stored["schema_version"], IDENTITY_SCHEMA_VERSION)

    def test_bad_identity_decision_is_400_and_existing_next_route_still_works(self) -> None:
        status, response = self.post_json(
            "/api/identity/labels",
            {
                "case_id": self.case["case_id"],
                "entity_id": "char-main",
                "decision": "maybe",
                "faces": [],
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("decision", response["error"])
        existing = self.get_json("/api/next")
        self.assertEqual(existing["case"]["case_id"], self.case["case_id"])

    def test_identity_page_script_and_frame_page_link_are_served(self) -> None:
        self.assertIn(
            "Qual destes rostos é este personagem?",
            self.get_text("/identity"),
        )
        self.assertIn("loadNext", self.get_text("/identity.js"))
        self.assertIn('href="/identity"', self.get_text("/"))

    def test_identity_face_markers_and_action_bar_styles_are_served(self) -> None:
        css = self.get_text("/style.css")
        face_marker = re.search(r"\.face-box\s*\{([^}]*)\}", css)
        action_bar = re.search(r"\.identity-decision-panel\s*\{([^}]*)\}", css)
        self.assertIsNotNone(face_marker)
        self.assertIn("background: transparent", face_marker.group(1))
        self.assertIsNotNone(action_bar)
        self.assertIn("position: sticky", action_bar.group(1))


if __name__ == "__main__":
    unittest.main()
