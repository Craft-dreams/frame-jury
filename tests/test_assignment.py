"""Tests for frame_jury.checks.assignment — Hungarian algorithm implementation."""

from __future__ import annotations

import itertools
import random
import unittest

from frame_jury.checks.assignment import best_assignment


def _brute_force_assignment(cost: list[list[float]]) -> tuple[float, list[tuple[int, int]]]:
    """Exhaustive search over all valid matchings to find maximum total similarity."""
    n_rows = len(cost)
    n_cols = len(cost[0])
    best_sum = -float("inf")
    best_pairs: list[tuple[int, int]] = []

    if n_rows <= n_cols:
        for p in itertools.permutations(range(n_cols), n_rows):
            s = sum(cost[i][p[i]] for i in range(n_rows))
            if s > best_sum:
                best_sum = s
                best_pairs = [(i, p[i]) for i in range(n_rows)]
    else:
        for p in itertools.permutations(range(n_rows), n_cols):
            s = sum(cost[p[j]][j] for j in range(n_cols))
            if s > best_sum:
                best_sum = s
                best_pairs = sorted([(p[j], j) for j in range(n_cols)])

    return best_sum, best_pairs


class TestAssignment(unittest.TestCase):
    """Unit tests for best_assignment."""

    def test_empty_cost_matrix(self) -> None:
        self.assertEqual(best_assignment([]), [])
        self.assertEqual(best_assignment([[]]), [])

    def test_single_element(self) -> None:
        self.assertEqual(best_assignment([[0.75]]), [(0, 0)])

    def test_greedy_vs_optimal_picks_higher_total(self) -> None:
        """Construct a case where greedy assignment and optimal assignment differ.

        Character 0: Face 0 = 0.90, Face 1 = 0.80
        Character 1: Face 0 = 0.85, Face 1 = 0.10

        Greedy:
          Char 0 picks Face 0 (0.90) -> Char 1 gets Face 1 (0.10). Total = 1.00.
        Optimal:
          Char 0 gets Face 1 (0.80) and Char 1 gets Face 0 (0.85). Total = 1.65.
        """
        cost = [
            [0.90, 0.80],
            [0.85, 0.10],
        ]
        pairs = best_assignment(cost)
        self.assertEqual(pairs, [(0, 1), (1, 0)])
        total_similarity = sum(cost[i][j] for i, j in pairs)
        self.assertAlmostEqual(total_similarity, 1.65)
        # Verify it strictly beats the greedy choice:
        greedy_total = cost[0][0] + cost[1][1]
        self.assertGreater(total_similarity, greedy_total)

    def test_rectangular_more_characters_than_faces(self) -> None:
        """When there are more characters than faces, surplus characters remain unassigned."""
        # 3 characters, 2 faces.
        cost = [
            [0.9, 0.1],
            [0.8, 0.85],
            [0.2, 0.3],
        ]
        pairs = best_assignment(cost)
        self.assertEqual(len(pairs), 2)
        # Optimal: Char 0 -> Face 0 (0.9), Char 1 -> Face 1 (0.85). Char 2 is surplus.
        self.assertEqual(pairs, [(0, 0), (1, 1)])

    def test_rectangular_more_faces_than_characters(self) -> None:
        """When there are more faces than characters, surplus faces remain unassigned."""
        # 2 characters, 3 faces.
        cost = [
            [0.9, 0.1, 0.4],
            [0.2, 0.85, 0.7],
        ]
        pairs = best_assignment(cost)
        self.assertEqual(len(pairs), 2)
        # Optimal: Char 0 -> Face 0 (0.9), Char 1 -> Face 1 (0.85). Face 2 is surplus.
        self.assertEqual(pairs, [(0, 0), (1, 1)])

    def test_negative_similarities(self) -> None:
        """Handles negative cosine similarities correctly."""
        cost = [
            [-0.8, -0.2],
            [-0.5, -0.9],
        ]
        # Matching (0, 1) [-0.2] and (1, 0) [-0.5] gives sum = -0.7.
        # Matching (0, 0) [-0.8] and (1, 1) [-0.9] gives sum = -1.7.
        pairs = best_assignment(cost)
        self.assertEqual(pairs, [(0, 1), (1, 0)])
        self.assertAlmostEqual(sum(cost[i][j] for i, j in pairs), -0.7)

    def test_random_matrices_against_brute_force(self) -> None:
        """Validate optimal assignment against brute force over 100 random rectangular instances."""
        rng = random.Random(1337)
        for _ in range(100):
            r = rng.randint(1, 4)
            c = rng.randint(1, 4)
            cost = [[rng.uniform(-1.0, 1.0) for _ in range(c)] for _ in range(r)]
            pairs = best_assignment(cost)
            self.assertEqual(len(pairs), min(r, c))

            # Verify no two pairs share the same row or column.
            rows = [i for i, _ in pairs]
            cols = [j for _, j in pairs]
            self.assertEqual(len(rows), len(set(rows)))
            self.assertEqual(len(cols), len(set(cols)))

            algo_total = sum(cost[i][j] for i, j in pairs)
            bf_total, _ = _brute_force_assignment(cost)
            self.assertAlmostEqual(algo_total, bf_total, places=6)


if __name__ == "__main__":
    unittest.main()
