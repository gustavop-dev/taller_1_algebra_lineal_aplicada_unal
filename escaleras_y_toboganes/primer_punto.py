"""
snakes_and_ladders_expected_rolls.py
------------------------------------

Utility for computing the expected number of dice rolls required to
finish a game of Snakes and Ladders (Escaleras y Toboganes) on a
configurable board.

Board specification
-------------------
The board is described with a JSON-compatible dictionary containing:

    * `length` – int  
      The index (1-based) of the final square.
    * `links` – list[tuple[int, int]]  
      Each tuple `(start, end)` represents a ladder (start < end) or a
      snake (start > end). Landing on `start` instantly moves the player
      to `end`.

Assumptions
-----------
* A fair six-sided die is used (faces 1–6).
* If a roll would move the token beyond the final square, it is placed
  **on** the final square (the game ends immediately).
* After moving according to the die, the link (if any) on the landing
  square is applied.

Algorithm
---------
Let E[i] be the expected throws needed to finish when the token is on
square i.  For the absorbing state (the last square) `E[length] = 0`.
For any other square:

    E[i] = 1 + (1/6) * Σ_{d=1}^{6} E[dest(i, d)]

where `dest(i, d)` is the square reached after rolling d and applying
the board's links.  This yields a linear system solved with NumPy.

Example
-------
>>> from snakes_and_ladders_expected_rolls import expected_rolls
>>> board = {"length": 9, "links": [(2, 4), (7, 3), (6, 8)]}
>>> expected_rolls(board)
6.264550264550265
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _build_transition_destinations(length: int,
                                   links: Dict[int, int]) -> List[List[int]]:
    """Return, for every square i (1-based), the list of destination
    squares obtained after rolling 1–6."""
    dests: List[List[int]] = [[] for _ in range(length + 1)]  # dummy index 0

    for i in range(1, length + 1):
        row: List[int] = []
        for die in range(1, 7):
            j = i + die
            if j >= length:
                row.append(length)
            else:
                row.append(links.get(j, j))
        dests[i] = row

    return dests[1:]  # drop dummy


def expected_rolls(board: Dict) -> float:
    """Compute the expected number of dice throws to finish the game.

    Parameters
    ----------
    board : dict
        Must contain:
            * 'length' – size of the board (index of final square, 1-based)
            * 'links'  – list of (start, end) pairs (ladders/snakes)

    Returns
    -------
    float
        Expected number of throws starting from square 1.
    """
    length: int = board["length"]
    links: Dict[int, int] = {start: end for start, end in board["links"]}

    n = length
    A = np.zeros((n, n))
    b = np.ones(n)

    # Terminal square equation: E[length] = 0
    A[n - 1, n - 1] = 1
    b[n - 1] = 0

    dests = _build_transition_destinations(length, links)

    for i in range(n - 1):          # skip the last (absorbing) state
        A[i, i] = 1
        for dest in dests[i]:
            A[i, dest - 1] -= 1 / 6

    E = np.linalg.solve(A, b)
    return float(E[0])


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python snakes_and_ladders_expected_rolls.py <board.json>",
              file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        board_def = json.load(f)

    print(f"Expected number of dice rolls: {expected_rolls(board_def):.6f}")
