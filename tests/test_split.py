from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.corpus.split import split_cases
from tests.helpers import make_case


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_never_leaks_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                make_case(root, "run-a", "shot-1"),
                make_case(root, "run-a", "shot-2"),
                make_case(root, "run-b", "shot-1"),
                make_case(root, "run-c", "shot-1"),
                make_case(root, "run-d", "shot-1"),
            ]

            first = split_cases(cases, seed=19, test_fraction=0.5)
            second = split_cases(list(reversed(cases)), seed=19, test_fraction=0.5)

            self.assertEqual(first, second)
            dev_runs = set(first["dev"]["run_ids"])
            test_runs = set(first["test"]["run_ids"])
            self.assertFalse(dev_runs & test_runs)
            self.assertEqual(dev_runs | test_runs, {"run-a", "run-b", "run-c", "run-d"})
            for case_id in first["dev"]["case_ids"]:
                self.assertIn(case_id.split("/", 1)[0], dev_runs)
            for case_id in first["test"]["case_ids"]:
                self.assertIn(case_id.split("/", 1)[0], test_runs)


if __name__ == "__main__":
    unittest.main()
