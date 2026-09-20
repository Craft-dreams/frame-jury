from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.corpus.schema import write_cases
from benchmarks.label.schema import LABEL_SCHEMA_VERSION, LabelValidationError, validate_label
from benchmarks.label.store import LabelStore
from tests.helpers import make_case


def label(case_id: str, defects: list[str]) -> dict[str, object]:
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "case_id": case_id,
        "defects": defects,
        "notes": "",
        "labeller": "tester",
        "at": "2026-09-20T12:00:00Z",
    }


class LabelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
