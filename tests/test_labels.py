from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.corpus.schema import write_cases
from benchmarks.label.schema import (
    DEFECTS,
    LABEL_SCHEMA_VERSION,
    TAXONOMY_VERSION,
    LabelValidationError,
    validate_label,
)
from benchmarks.label.store import LabelStore, queue_order
from tests.helpers import make_case


def label(case_id: str, defects: list[str]) -> dict[str, object]:
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "case_id": case_id,
        "defects": defects,
        "notes": "",
        "labeller": "tester",
        "at": "2026-09-20T12:00:00Z",
    }


class LabelTests(unittest.TestCase):
    def test_taxonomy_shortcut_order_has_ten_current_defects(self) -> None:
        self.assertEqual(
            DEFECTS,
            (
                "duplicated_character",
                "broken_hands",
                "extra_person",
                "missing_entity",
                "wrong_identity",
                "broken_body",
                "wrong_scale",
                "fused_objects",
                "garbled_text",
                "empty_or_flat",
            ),
        )

    def test_current_labels_name_the_taxonomy_and_legacy_anatomy_remains_valid(self) -> None:
        current = label("run-a/shot-1", ["broken_hands", "wrong_scale"])
        self.assertEqual(validate_label(current)["taxonomy_version"], TAXONOMY_VERSION)

        legacy = current.copy()
        legacy.pop("taxonomy_version")
        legacy["schema_version"] = "1.0"
        legacy["defects"] = ["broken_anatomy"]
        self.assertEqual(validate_label(legacy)["defects"], ["broken_anatomy"])

    def test_label_schema_rejects_mixed_exclusive_decisions(self) -> None:
        with self.assertRaisesRegex(LabelValidationError, "exclusive"):
            validate_label(label("run-a/shot-1", ["clean", "missing_entity"]))
        with self.assertRaisesRegex(LabelValidationError, "timezone"):
            invalid = label("run-a/shot-1", ["clean"])
            invalid["at"] = "2026-09-20T12:00:00"
            validate_label(invalid)

    def test_store_appends_and_reopening_resumes_at_first_unlabelled_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            labels_path = root / "labels.jsonl"
            cases = [make_case(root, "run-a", "shot-1"), make_case(root, "run-b", "shot-1")]
            write_cases(cases_path, cases)

            store = LabelStore(cases_path, labels_path)
            self.assertEqual(store.next_case()["case_id"], "run-a/shot-1")
            store.append(label("run-a/shot-1", ["missing_entity", "garbled_text"]))

            resumed = LabelStore(cases_path, labels_path)
            self.assertEqual(resumed.labelled_count, 1)
            self.assertEqual(resumed.positive_counts["missing_entity"], 1)
            self.assertEqual(resumed.positive_counts["garbled_text"], 1)
            self.assertEqual(resumed.next_case()["case_id"], "run-b/shot-1")
            resumed.append(label("run-b/shot-1", ["clean"]))
            lines = labels_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2, "saving a second label must append, not overwrite")
            self.assertEqual(json.loads(lines[0])["case_id"], "run-a/shot-1")
            self.assertIsNone(LabelStore(cases_path, labels_path).next_case())

    def test_store_refuses_a_second_decision_for_the_same_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            labels_path = root / "labels.jsonl"
            write_cases(cases_path, [make_case(root, "run-a", "shot-1")])
            store = LabelStore(cases_path, labels_path)
            store.append(label("run-a/shot-1", ["uncertain"]))
            with self.assertRaisesRegex(LabelValidationError, "already labelled"):
                store.append(label("run-a/shot-1", ["clean"]))

    def test_store_counts_legacy_anatomy_without_reinterpreting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            labels_path = root / "labels.jsonl"
            write_cases(cases_path, [make_case(root, "run-a", "shot-1")])
            legacy = label("run-a/shot-1", ["broken_hands"])
            legacy.pop("taxonomy_version")
            legacy["schema_version"] = "1.0"
            legacy["defects"] = ["broken_anatomy"]
            labels_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

            store = LabelStore(cases_path, labels_path)

            self.assertEqual(store.legacy_broken_anatomy_count, 1)
            self.assertEqual(store.positive_counts["broken_hands"], 0)
            self.assertEqual(store.positive_counts["broken_body"], 0)

    def test_page_leads_with_declaration_and_keeps_prompts_visible(self) -> None:
        static = Path(__file__).parents[1] / "benchmarks" / "label" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        javascript = (static / "app.js").read_text(encoding="utf-8")

        ordered_markers = [
            'id="framing"',
            "Must be visible",
            'id="entities"',
            "No character is declared in this shot — only the setting",
            "Staging",
            "Positive prompt",
            "Negative prompt",
        ]
        positions = [html.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("entity.display_name", javascript)
        self.assertIn("entity.visual_identity", javascript)
        self.assertIn("entity.relative_scale", javascript)
        self.assertIn('const shortcuts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]', javascript)

    def test_queue_prioritizes_character_count_then_framing_and_groups_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def case(run_id: str, shot_id: str, characters: int, framing: str) -> dict[str, object]:
                value = make_case(root, run_id, shot_id)
                value["shot"]["framing"] = framing
                value["shot"]["declared_entities"] = [
                    {
                        "entity_id": f"char-{index}",
                        "kind": "character",
                        "display_name": f"Character {index}",
                        "aliases": [],
                        "reference_images": [],
                    }
                    for index in range(characters)
                ]
                value["reference_images"] = {
                    entity["entity_id"]: []
                    for entity in value["shot"]["declared_entities"]
                }
                return value

            cases = [
                case("run-b", "zero-close", 0, "close-up"),
                case("run-b", "two-wide", 2, "wide shot"),
                case("run-a", "one-wide", 1, "wide shot"),
                case("run-a", "two-close", 2, "medium shot"),
                case("run-a", "two-close-2", 2, "close-up"),
            ]
            ordered = [item["case_id"] for item in sorted(cases, key=queue_order)]
            self.assertEqual(
                ordered,
                [
                    "run-a/two-close",
                    "run-a/two-close-2",
                    "run-b/two-wide",
                    "run-a/one-wide",
                    "run-b/zero-close",
                ],
            )


if __name__ == "__main__":
    unittest.main()
