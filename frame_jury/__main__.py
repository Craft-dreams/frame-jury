"""frame_jury.__main__ — ``python -m frame_jury judge``, the batch command line.

SPEC.md §10: the factory talks to frame-jury only through the JSON contract of
§4, in a separate process (frame-jury is AGPL; nothing in the factory may
import it).  Until now frame-jury was only a library
(``frame_jury.jury.judge``), so a caller paid the backend load — torch, the
weights, OpenCV — once per frame.  This command runs one process over a whole
batch: the detector, face and VLM backends are constructed once, lazily, and
only when some request in the batch needs them.

Usage::

    python -m frame_jury judge --requests REQUESTS.json --out VERDICTS.json

REQUESTS.json holds a JSON list of request objects.  Each is exactly the §4
request plus one caller-chosen string ``request_id`` — stripped before
``JuryRequest.from_dict``, because it is not part of the contract.  Duplicate
or missing ``request_id`` values are a usage error: exit 1 with a message on
stderr and nothing judged.

VERDICTS.json holds a JSON list, in request order, of
``{"request_id": id, "verdict": …}`` or, for a request that failed to parse or
whose judgement raised, ``{"request_id": id, "error": "<Type>: <message>"}``.
The list is written to ``<out>.tmp`` and replaced after every request, so a
crash keeps the frames already judged.

Exit codes: 0 when every request produced a verdict, 2 when at least one
errored, 1 on usage errors.  stderr carries one progress line per frame.

No change to ``judge``, the contract, thresholds or backends: §4 stays the
whole surface.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from frame_jury import jury
from frame_jury.contract import VALID_CHECKS, JuryRequest

# The CLI routes on the same check names the contract accepts; an empty or
# omitted ``checks`` list means every check, exactly as ``judge`` reads it.
_DEFAULT_CHECKS: frozenset[str] = frozenset(VALID_CHECKS)


class _UsageError(Exception):
    """Bad arguments or a malformed request file: the batch exits 1."""


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of exiting on a usage error.

    argparse's default ``error`` exits with code 2, which the contract reserves
    for a batch where at least one frame errored; usage errors must exit 1.
    """

    def error(self, message: str) -> None:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="python -m frame_jury",
        description="Judge generated frames against the declarations that asked for them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    judge_parser = subparsers.add_parser(
        "judge",
        help="judge a batch of frames; the backends load once, not once per frame",
        description="Judge a batch of frames; the backends load once, not once per frame.",
    )
    judge_parser.add_argument(
        "--requests",
        required=True,
        type=Path,
        help="JSON list of §4 requests, each plus a caller-chosen string 'request_id'",
    )
    judge_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="verdicts are written here, in request order",
    )
    return parser


class _BackendPool:
    """Builds each heavy backend once for the whole batch, on first need.

    The reason this command exists: ``judge`` called with ``detector=None``
    resolves a fresh default backend for every frame.  Here the resolver runs
    the first time a request needs the backend, and every later request shares
    the built one.  A failing build is cached too, so a broken environment
    fails each affected request with the same error instead of retrying the
    load frame after frame.
    """

    _RESOLVER_NAMES = {
        "detector": "_resolve_detector",
        "face_backend": "_resolve_face_backend",
        "vlm_scorer": "_resolve_vlm_scorer",
    }

    def __init__(self) -> None:
        self._built: dict[str, Any] = {}

    def build(self, kind: str) -> Any:
        """Return the shared backend for *kind*, or raise its cached failure."""
        if kind not in self._built:
            resolver = getattr(jury, self._RESOLVER_NAMES[kind])
            try:
                self._built[kind] = resolver(None)
            except Exception as exc:  # noqa: BLE001 — one broken backend must not stop the batch
                self._built[kind] = exc
        built = self._built[kind]
        if isinstance(built, BaseException):
            raise built
        return built


def _judge_one(
    backends: _BackendPool, request_id: str, raw_request: dict[str, Any]
) -> dict[str, Any]:
    """Parse and judge one raw request; never raises.

    Which backends to build mirrors how ``judge`` routes: presence needs the
    detector, identity the face backend, and the VLM only a request that both
    asks for ``vlm_scene`` and runs under ``budget: "full"``.  An omitted or
    empty ``checks`` list requests every check, as in ``judge``.  Any failure —
    a contract error on parse, a backend that will not load, or an error from
    ``judge`` itself — becomes an ``error`` entry and the batch continues.
    """
    try:
        request = JuryRequest.from_dict(raw_request)
        requested = frozenset(request.checks) if request.checks else _DEFAULT_CHECKS
        verdict = jury.judge(
            request,
            detector=backends.build("detector") if "presence" in requested else None,
            face_backend=backends.build("face_backend") if "identity" in requested else None,
            vlm_scorer=(
                backends.build("vlm_scorer")
                if "vlm_scene" in requested and request.budget == "full"
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 — one bad frame must not stop the batch
        return {"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"}
    return {"request_id": request_id, "verdict": verdict.to_dict()}


def _write_output(out_path: Path, results: list[dict[str, Any]]) -> None:
    """Rewrite the output file atomically after every request.

    The list is written to ``<out>.tmp`` then replaced, so a crash mid-batch
    keeps every frame already judged.  The temporary file lives beside the
    output, on the same volume, so the replacement is atomic.
    """
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(out_path)


def _load_requests(requests_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read and validate the request file; pair every id with its §4 request.

    ``request_id`` is the spine of the output — every verdict and error is
    reported under it — so a missing, non-string or duplicate id is a usage
    error for the whole batch: exit 1, nothing judged.
    """
    try:
        text = requests_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _UsageError(f"cannot read requests file: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _UsageError(f"requests file {requests_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise _UsageError(f"requests file {requests_path} must contain a JSON list of request objects")

    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _UsageError(f"request {index}: must be a JSON object")
        if "request_id" not in item:
            raise _UsageError(f"request {index}: missing 'request_id'")
        request_id = item["request_id"]
        if not isinstance(request_id, str):
            raise _UsageError(f"request {index}: 'request_id' must be a string")
        if request_id in seen:
            raise _UsageError(f"duplicate 'request_id': {request_id!r}")
        seen.add(request_id)
        stripped = {key: value for key, value in item.items() if key != "request_id"}
        entries.append((request_id, stripped))
    return entries


def _run_judge(requests_path: Path, out_path: Path) -> int:
    """Judge every request in *requests_path*; return the process exit code."""
    entries = _load_requests(requests_path)
    backends = _BackendPool()
    results: list[dict[str, Any]] = []

    if not entries:
        _write_output(out_path, results)  # an empty batch still writes an empty list

    errored = False
    for index, (request_id, raw_request) in enumerate(entries, start=1):
        started = time.perf_counter()
        result = _judge_one(backends, request_id, raw_request)
        elapsed = time.perf_counter() - started

        results.append(result)
        _write_output(out_path, results)

        verdict = result.get("verdict", {})
        status = "error" if result.get("error") else verdict["verdict"]
        findings = len(verdict.get("findings", [])) or 0
        abstentions = len(verdict.get("abstentions", [])) or 0
        errored = errored or status == "error"
        print(
            f"judged {index}/{len(entries)} {request_id}: {status} "
            f"({findings} findings, {abstentions} abstentions) in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return 2 if errored else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    try:
        args = _build_parser().parse_args(argv)
        if args.command != "judge":
            raise _UsageError(f"unknown command: {args.command!r}")
        return _run_judge(args.requests, args.out)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # An unwritable --out (or unreadable --requests) is a usage error.
        # Failures on a frame's own image path are reported per request
        # in the output list and never land here.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
