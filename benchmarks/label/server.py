"""Serve the local keyboard-first frame labelling interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlparse

from frame_jury.backends.base import FaceBackend

from .faces import crop, face_boxes
from .identity import (
    IDENTITY_SCHEMA_VERSION,
    IdentityLabelError,
    IdentityStore,
)
from .schema import (
    DEFECTS,
    LABEL_SCHEMA_VERSION,
    TAXONOMY_VERSION,
    LabelValidationError,
)
from .store import LabelStore

_STATIC_DIR = Path(__file__).with_name("static")


def _public_case(case: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(case))
    case_id = quote(case["case_id"], safe="")
    public["image_url"] = f"/asset?case_id={case_id}&kind=frame"
    for entity in public["shot"]["declared_entities"]:
        entity_id = quote(entity["entity_id"], safe="")
        entity["reference_urls"] = [
            f"/asset?case_id={case_id}&kind=reference&entity_id={entity_id}&index={index}"
            for index, _path in enumerate(entity["reference_images"])
        ]
    return public


def make_handler(
    store: LabelStore,
    labeller: str,
    *,
    identity_store: IdentityStore | None = None,
    face_backend_factory: Callable[[], FaceBackend] | None = None,
) -> type[BaseHTTPRequestHandler]:
    resolved_face_backend: FaceBackend | None = None
    backend_lock = threading.Lock()

    def get_face_backend() -> FaceBackend:
        nonlocal resolved_face_backend
        if resolved_face_backend is None:
            with backend_lock:
                if resolved_face_backend is None:
                    if face_backend_factory is None:
                        from frame_jury.jury import _resolve_face_backend

                        resolved_face_backend = _resolve_face_backend(None)
                    else:
                        resolved_face_backend = face_backend_factory()
        return resolved_face_backend

    class Handler(BaseHTTPRequestHandler):
        server_version = "frame-jury-label/2.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _json(self, status: HTTPStatus, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, filename: str) -> None:
            path = _STATIC_DIR / filename
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _asset(self, query: dict[str, list[str]]) -> None:
            case_id = query.get("case_id", [""])[0]
            case = store.case(case_id)
            if case is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            kind = query.get("kind", [""])[0]
            if kind == "frame":
                path = Path(case["image_path"])
            elif kind == "reference":
                entity_id = query.get("entity_id", [""])[0]
                try:
                    index = int(query.get("index", [""])[0])
                    path = Path(case["reference_images"][entity_id][index])
                except (KeyError, IndexError, ValueError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            elif kind in {"frame_face", "reference_face"}:
                try:
                    index = int(query.get("index", [""])[0])
                    if kind == "frame_face":
                        path = Path(case["image_path"])
                    else:
                        entity_id = query.get("entity_id", [""])[0]
                        reference_index = int(query.get("ref", [""])[0])
                        path = Path(case["reference_images"][entity_id][reference_index])
                    boxes = face_boxes(str(path), get_face_backend())
                    if index < 0:
                        raise IndexError
                    body = crop(str(path), boxes[index])
                except (KeyError, IndexError, ValueError, OSError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._static("index.html")
            elif parsed.path == "/identity":
                self._static("identity.html")
            elif parsed.path == "/app.js":
                self._static("app.js")
            elif parsed.path == "/identity.js":
                self._static("identity.js")
            elif parsed.path == "/style.css":
                self._static("style.css")
            elif parsed.path == "/api/next":
                case = store.next_case()
                self._json(
                    HTTPStatus.OK,
                    {
                        "case": _public_case(case) if case else None,
                        "defects": DEFECTS,
                        "labeller": labeller,
                        "labelled": store.labelled_count,
                        "total": store.total_count,
                        "positive_counts": store.positive_counts,
                        "legacy_broken_anatomy": store.legacy_broken_anatomy_count,
                    },
                )
            elif parsed.path == "/api/identity/next":
                if identity_store is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                pair = identity_store.next_pair()
                response: dict[str, Any] = {
                    "case": None,
                    "entity_id": None,
                    "entity": None,
                    "frame_faces": [],
                    "reference_faces": [],
                    "labelled": identity_store.labelled_count,
                    "total": identity_store.total_count,
                    "labeller": labeller,
                }
                if pair is not None:
                    case, entity_id = pair
                    public_case = _public_case(case)
                    entity = next(
                        item
                        for item in public_case["shot"]["declared_entities"]
                        if item["entity_id"] == entity_id
                    )
                    backend = get_face_backend()
                    frame = str(case["image_path"])
                    reference = str(case["reference_images"][entity_id][0])
                    case_query = quote(case["case_id"], safe="")
                    entity_query = quote(entity_id, safe="")
                    response.update(
                        {
                            "case": public_case,
                            "entity_id": entity_id,
                            "entity": entity,
                            "frame_faces": [
                                {
                                    "index": index,
                                    "box": box,
                                    "url": (
                                        f"/asset?case_id={case_query}&kind=frame_face"
                                        f"&index={index}"
                                    ),
                                }
                                for index, box in enumerate(face_boxes(frame, backend))
                            ],
                            "reference_faces": [
                                {
                                    "index": index,
                                    "box": box,
                                    "url": (
                                        f"/asset?case_id={case_query}&kind=reference_face"
                                        f"&entity_id={entity_query}&ref=0&index={index}"
                                    ),
                                }
                                for index, box in enumerate(face_boxes(reference, backend))
                            ],
                        }
                    )
                self._json(HTTPStatus.OK, response)
            elif parsed.path == "/asset":
                self._asset(parse_qs(parsed.query))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            if path not in {"/api/labels", "/api/identity/labels"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                if path == "/api/labels":
                    label = {
                        "schema_version": LABEL_SCHEMA_VERSION,
                        "taxonomy_version": TAXONOMY_VERSION,
                        "case_id": payload.get("case_id"),
                        "defects": payload.get("defects"),
                        "notes": payload.get("notes", ""),
                        "labeller": labeller,
                        "at": now,
                    }
                    store.append(label)
                else:
                    if identity_store is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    case_id = payload.get("case_id")
                    case = identity_store.case(case_id) if isinstance(case_id, str) else None
                    if case is None:
                        raise IdentityLabelError(f"case_id is unknown: {case_id!r}")
                    record = {
                        "schema_version": IDENTITY_SCHEMA_VERSION,
                        "case_id": case_id,
                        "entity_id": payload.get("entity_id"),
                        "decision": payload.get("decision"),
                        "faces": payload.get("faces"),
                        "face_boxes": face_boxes(str(case["image_path"]), get_face_backend()),
                        "labeller": labeller,
                        "at": now,
                    }
                    identity_store.append(record)
            except (
                json.JSONDecodeError,
                ValueError,
                LabelValidationError,
                IdentityLabelError,
            ) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.CREATED, {"ok": True})

    return Handler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument(
        "--labels", type=Path, default=Path("benchmarks/labels/labels.jsonl")
    )
    parser.add_argument(
        "--identity-labels",
        type=Path,
        default=Path("benchmarks/labels/identity.jsonl"),
    )
    parser.add_argument("--labeller", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = LabelStore(args.cases, args.labels)
        identity_store = IdentityStore(
            args.cases,
            args.identity_labels,
            frame_labels_path=args.labels,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(store, args.labeller, identity_store=identity_store),
    )
    url = f"http://{args.host}:{server.server_port}/"
    print(
        f"Labelling {store.total_count - store.labelled_count} remaining of "
        f"{store.total_count} cases at {url} (Ctrl+C to stop)"
    )
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
