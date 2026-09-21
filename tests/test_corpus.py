from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from benchmarks.corpus.build import build_corpus
from benchmarks.corpus.schema import CaseValidationError, validate_case, write_cases
from tests.helpers import make_case, make_run, write_png


class CorpusBuilderTests(unittest.TestCase):
    def test_joins_frame_declaration_world_references_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            make_run(root)

            cases, report = build_corpus(root)

            self.assertEqual(report.runs_seen, 1)
            self.assertEqual(report.runs_usable, 1)
            self.assertEqual(report.cases_emitted, 1)
            self.assertFalse(report.cases_skipped_by_reason)
            case = cases[0]
            self.assertEqual(case["case_id"], "run-example/shot-scene-001-001")
            self.assertTrue(Path(case["image_path"]).is_absolute())
            self.assertEqual(case["shot"]["framing"], "close-up")
            self.assertEqual(
                case["shot"]["staging"],
                {
                    "purpose": "The keeper leans over the notebook.",
                    "must_render": ["The notebook must be open."],
                    "composition": ["Keep both entities in focus."],
                },
            )
            self.assertEqual(case["shot"]["positive_prompt"], "A tired keeper bends over an open notebook.")
            self.assertEqual(case["shot"]["negative_prompt"], "extra people, duplicate keeper")
            self.assertEqual(
                [
                    (
                        entity["entity_id"],
                        entity["display_name"],
                        entity["aliases"],
                        entity["visual_identity"],
                        entity["relative_scale"],
                        entity["approximate_dimensions"],
                    )
                    for entity in case["shot"]["declared_entities"]
                ],
                [
                    (
                        "char-keeper",
                        "The keeper",
                        ["night watchman"],
                        "A tired keeper in a dark wool coat.",
                        "adult human",
                        "1.75 m tall",
                    ),
                    (
                        "obj-notebook",
                        "Open notebook",
                        [],
                        "A worn, cloth-bound notebook.",
                        "hand-held",
                        "20 cm by 14 cm",
                    ),
                ],
            )
            self.assertEqual(case["provenance"], {"run_id": "run-example", "model": "fixture-model", "seed": 1234})
            self.assertEqual(cases, build_corpus(root)[0], "the same input must preserve order and bytes")

    def test_missing_stage_skips_whole_run_without_writing_to_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = make_run(root)
            reference_dir = run / "05-production-bible" / "reference-sheet"
            for path in reference_dir.iterdir():
                path.unlink()
            reference_dir.rmdir()
            before = sorted(path.relative_to(run) for path in run.rglob("*"))

            cases, report = build_corpus(root)

            after = sorted(path.relative_to(run) for path in run.rglob("*"))
            self.assertEqual(cases, [])
            self.assertEqual(report.runs_usable, 0)
            self.assertEqual(report.runs_skipped_by_reason["missing_reference_sheet"], 1)
            self.assertEqual(report.cases_skipped_by_reason["missing_reference_sheet"], 1)
            self.assertEqual(before, after, "the borrowed run must remain untouched")

    def test_malformed_declaration_is_counted_and_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = make_run(root)
            direction_path = run / "07-direction" / "outcome.json"
            import json

            direction = json.loads(direction_path.read_text(encoding="utf-8"))
            direction["beats"][0]["scene_plan"]["shot_specs"][0]["required_visible_entity_ids"] = "char-keeper"
            direction_path.write_text(json.dumps(direction), encoding="utf-8")

            cases, report = build_corpus(root)

            self.assertEqual(cases, [])
            self.assertEqual(report.runs_usable, 0)
            self.assertEqual(report.cases_skipped_by_reason["malformed_declaration"], 1)
            self.assertEqual(report.runs_skipped_by_reason["no_emitted_cases"], 1)

    def test_bible_profiles_are_optional_as_a_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            run = make_run(root)
            (run / "05-production-bible" / "bible.json").unlink()

            cases, report = build_corpus(root)

            self.assertEqual(report.runs_usable, 1)
            for entity in cases[0]["shot"]["declared_entities"]:
                self.assertNotIn("visual_identity", entity)
                self.assertNotIn("relative_scale", entity)
                self.assertNotIn("approximate_dimensions", entity)


class CaseSchemaTests(unittest.TestCase):
    def test_malformed_case_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = make_case(Path(directory), "run-a", "shot-a")
            case["provenance"]["seed"] = "seven"
            with self.assertRaisesRegex(CaseValidationError, "seed"):
                validate_case(case)

    def test_writer_rejects_duplicate_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = make_case(root, "run-a", "shot-a")
            output = root / "cases.jsonl"
            write_cases(output, [case])
            self.assertIn("run-a/shot-a", output.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(CaseValidationError, "duplicate case_id"):
                write_cases(output, [case, copy.deepcopy(case)])
            self.assertIn(
                "run-a/shot-a",
                output.read_text(encoding="utf-8"),
                "a failed atomic rewrite must preserve the existing valid corpus",
            )

    def test_visual_profile_fields_must_appear_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = make_case(Path(directory), "run-a", "shot-a")
            entity = {
                "entity_id": "char-a",
                "kind": "character",
                "display_name": "A",
                "aliases": [],
                "reference_images": [],
                "visual_identity": "A person.",
            }
            case["shot"]["declared_entities"] = [entity]
            case["reference_images"] = {"char-a": []}
            with self.assertRaisesRegex(CaseValidationError, "must appear together"):
                validate_case(case)


if __name__ == "__main__":
    unittest.main()
