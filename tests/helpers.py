from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from benchmarks.corpus.schema import CASE_SCHEMA_VERSION


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_png(path: Path, *, red: int = 0, green: int = 0, blue: int = 0) -> None:
    """Generate a valid 1x1 RGB PNG using only the standard library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.parent.mkdir(parents=True, exist_ok=True)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes((0, red, green, blue)))
    path.write_bytes(signature + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b""))


def make_run(root: Path, name: str = "run-example") -> Path:
    run = root / name
    shot_id = "shot-scene-001-001"
    direction = {
        "beats": [
            {
                "scene_plan": {
                    "scene_spec": {"scene_id": "scene-001"},
                    "shot_specs": [
                        {
                            "shot_id": shot_id,
                            "scene_id": "scene-001",
                            "purpose": "The keeper leans over the notebook.",
                            "must_render_constraints": ["The notebook must be open."],
                            "composition_constraints": ["Keep both entities in focus."],
                            "camera": {"framing": "close-up"},
                            "required_visible_entity_ids": ["char-keeper", "obj-notebook"],
                        }
                    ],
                }
            }
        ]
    }
    world = {
        "identities": [
            {
                "entity_id": "char-keeper",
                "kind": "character",
                "display_name": "The keeper",
                "aliases": ["night watchman"],
            },
            {
                "entity_id": "obj-notebook",
                "kind": "object",
                "display_name": "Open notebook",
                "aliases": [],
            },
        ]
    }
    write_json(run / "07-direction" / "outcome.json", direction)
    write_json(run / "03-world" / "canonical-world.json", world)
    write_json(
        run / "05-production-bible" / "bible.json",
        {
            "visual_profiles": [
                {
                    "entity_id": "char-keeper",
                    "visual_identity": "A tired keeper in a dark wool coat.",
                    "relative_scale": "adult human",
                    "approximate_dimensions": "1.75 m tall",
                },
                {
                    "entity_id": "obj-notebook",
                    "visual_identity": "A worn, cloth-bound notebook.",
                    "relative_scale": "hand-held",
                    "approximate_dimensions": "20 cm by 14 cm",
                },
            ]
        },
    )
    write_png(run / "05-production-bible" / "reference-sheet" / "char-keeper.png", red=255)
    write_png(run / "05-production-bible" / "reference-sheet" / "obj-notebook.png", green=255)
    image = run / "10-resolved-media" / "visuals" / f"{shot_id}.png"
    write_png(image, blue=255)
    write_json(
        image.with_name(f"{shot_id}.visual.json"),
        {
            "metadata": {
                "source_shot_id": shot_id,
                "prompt": "A tired keeper bends over an open notebook.",
                "negative_prompt": "extra people, duplicate keeper",
            },
            "model": "fixture-model",
            "seed": 1234,
        },
    )
    return run


def make_case(root: Path, run_id: str, shot_id: str) -> dict[str, Any]:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": f"{run_id}/{shot_id}",
        "image_path": str((root / run_id / f"{shot_id}.png").resolve()),
        "shot": {
            "shot_id": shot_id,
            "framing": "wide shot",
            "declared_entities": [],
            "staging": {
                "purpose": "An empty room.",
                "must_render": [],
                "composition": [],
            },
            "positive_prompt": "An empty room in a wide shot.",
            "negative_prompt": "people",
        },
        "reference_images": {},
        "provenance": {"run_id": run_id, "model": "fixture", "seed": 7},
    }
