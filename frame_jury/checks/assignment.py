"""frame_jury.checks.assignment — optimal bipartite matching via Hungarian algorithm.

Implements the Hungarian algorithm (Kuhn-Munkres algorithm) in pure Python
for optimal one-to-one assignment maximizing total similarity between declared
characters and detected faces.

Citations:
  - Harold W. Kuhn, "The Hungarian Method for the assignment problem",
    Naval Research Logistics Quarterly, 2:83–97, 1955.
  - James Munkres, "Algorithms for the Assignment and Transportation Problems",
    Journal of the Society for Industrial and Applied Mathematics, 5(1):32–38, 1957.
"""

from __future__ import annotations


def _hungarian_min_cost(matrix: list[list[float]]) -> list[int]:
    """Find a minimum-cost matching for an N x N matrix using the Kuhn-Munkres algorithm.

    Parameters
    ----------
    matrix:
        Square N x N cost matrix.

    Returns
    -------
    list[int]
        col_for_row mapping, where result[i] is the column index assigned to row i.
    """
    n = len(matrix)
    if n == 0:
        return []

    # 1-indexed potential and matching arrays for augmenting path search (O(N^3)).
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)      # p[j] is the row assigned to column j (1..n)
    way = [0] * (n + 1)    # way[j] records the column from which slack was updated

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    col_for_row = [0] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            col_for_row[p[j] - 1] = j - 1
    return col_for_row


def best_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment maximising total similarity.

    cost[i][j] is the similarity between declared character i and face j.
    Returns the chosen (i, j) pairs. Handles rectangular input: when there
    are more faces than characters, or more characters than faces, the
    surplus is simply left unassigned.
    """
    if not cost or not cost[0]:
        return []

    n_rows = len(cost)
    n_cols = len(cost[0])
    N = max(n_rows, n_cols)

    # To maximize similarity, convert to a non-negative minimization cost matrix.
    # C[i][j] = M - cost[i][j], where M >= max(cost).
    # For dummy rows and columns, setting similarity = 0.0 gives cost M.
    # In any bijection of the N x N padded matrix, exactly min(n_rows, n_cols) real
    # elements are matched, while dummy elements contribute a constant cost (N - min)*M.
    # Therefore, minimizing total cost is mathematically identical to maximizing
    # the sum of the matched real similarities.
    max_val = max(max(row) for row in cost)
    M = max(0.0, max_val)

    padded_matrix = [[M for _ in range(N)] for _ in range(N)]
    for i in range(n_rows):
        for j in range(n_cols):
            padded_matrix[i][j] = M - cost[i][j]

    col_for_row = _hungarian_min_cost(padded_matrix)

    pairs: list[tuple[int, int]] = []
    for i in range(n_rows):
        j = col_for_row[i]
        if j < n_cols:
            pairs.append((i, j))

    return sorted(pairs, key=lambda pair: pair[0])
