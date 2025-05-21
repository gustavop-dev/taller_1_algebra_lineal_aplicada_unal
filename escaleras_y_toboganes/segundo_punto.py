"""
snakes_and_ladders_expected_rolls.py
------------------------------------

Compute the expected number of dice rolls required to finish a game of
Snakes and Ladders (Escaleras y Toboganes) on a configurable board.

NEW: the die can be biased.  Pass the face–probability vector
`probs = (p1, …, p6)`; each pᵢ is the probability of rolling face *i*.
If omitted, a fair die (pᵢ = 1/6) is assumed.
"""

from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np


def _build_destinations(length: int,
                        links: Dict[int, int]) -> List[List[int]]:
    """For every square i (1-based), return the six destination squares
    reached after rolling 1 … 6 and applying any ladder/snake."""
    dests: List[List[int]] = [[] for _ in range(length + 1)]  # dummy 0
    for i in range(1, length + 1):
        row: List[int] = []
        for d in range(1, 7):
            j = i + d
            row.append(length if j >= length else links.get(j, j))
        dests[i] = row
    return dests[1:]


def expected_rolls(board: Dict,
                   probs: Sequence[float] = (1/6,)*6) -> float:
    """Expected number of throws starting from square 1.

    Parameters
    ----------
    board : dict
        Must contain:
            * 'length' – int, index of the final square (1-based).
            * 'links'  – list[tuple[int,int]] of ladders/snakes.
    probs : sequence of 6 floats, optional
        Face-probabilities (p₁, …, p₆).  Must sum to 1.

    Returns
    -------
    float
    """
    if len(probs) != 6 or not np.isclose(sum(probs), 1.0):
        raise ValueError("`probs` must be six numbers summing to 1.")

    length: int = board["length"]
    links: Dict[int, int] = {s: e for s, e in board["links"]}

    n = length
    A = np.zeros((n, n))
    b = np.ones(n)

    # Absorbing state: E[length] = 0
    A[n - 1, n - 1] = 1
    b[n - 1] = 0

    dests = _build_destinations(length, links)

    for i in range(n - 1):          # all non-terminal squares
        A[i, i] = 1
        for face, dest in enumerate(dests[i], start=1):
            A[i, dest - 1] -= probs[face - 1]

    E = np.linalg.solve(A, b)
    return float(E[0])


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) not in (2, 3):
        print("Usage: python snakes_and_ladders_expected_rolls.py "
              "<board.json> [p1,p2,p3,p4,p5,p6]", file=sys.stderr)
        sys.exit(1)

    board_def = json.load(open(sys.argv[1], encoding="utf-8"))

    if len(sys.argv) == 3:
        probs = tuple(map(float, sys.argv[2].split(",")))
        print(f"Expected rolls: {expected_rolls(board_def, probs):.6f}")
    else:
        print(f"Expected rolls: {expected_rolls(board_def):.6f}")
