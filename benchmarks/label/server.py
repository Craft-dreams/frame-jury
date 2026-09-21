"""Serve the local keyboard-first frame labelling interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

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


def make_handler(store: LabelStore, labeller: str) -> type[BaseHTTPRequestHandler]:
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
            elif parsed.path == "/app.js":
                self._static("app.js")
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
            elif parsed.path == "/asset":
                self._asset(parse_qs(parsed.query))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlparse(self.path).path != "/api/labels":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                label = {
                    "schema_version": LABEL_SCHEMA_VERSION,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "case_id": payload.get("case_id"),
                    "defects": payload.get("defects"),
                    "notes": payload.get("notes", ""),
                    "labeller": labeller,
                    "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                store.append(label)
            except (json.JSONDecodeError, ValueError, LabelValidationError) as exc:
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
    parser.add_argument("--labeller", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = LabelStore(args.cases, args.labels)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.labeller))
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
