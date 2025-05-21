"""
optimal_die.py
==============

Point 4 – “Find the optimal distribution on the die for any given board.”

The module minimises the expected number of rolls required to finish
a Snakes-and-Ladders board, subject to:

    • Σ p_i = 1                 (valid probability vector)
    • Σ i·p_i = 3.5             (same mean advance as a fair die)
    • p_i ≥ 0                   (non-negativity)

It re-uses the expected-rolls engine (Point 2) and calls SciPy’s SLSQP
optimizer to search the continuous space of feasible distributions.

----------------------------------------------------------------------
Usage
----------------------------------------------------------------------
>>> from optimal_die import optimise_die
>>> board = {"length": 9, "links": [(2, 4), (7, 3), (6, 8)]}
>>> p_opt, exp_rolls = optimise_die(board)
>>> print("Optimal p:", [round(x, 4) for x in p_opt])
>>> print("Expected rolls:", exp_rolls)

Requirements:  numpy, scipy   (pip install numpy scipy)
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize


# ────────────────────────────────────────────────────────────────────
# Expected-rolls function (from Point 2)
# ────────────────────────────────────────────────────────────────────
def _destinations(length: int, links: Dict[int, int]) -> List[List[int]]:
    """Return destination squares after rolling 1–6 from each square."""
    dest = [[] for _ in range(length + 1)]  # dummy zero index
    for i in range(1, length + 1):
        row = []
        for d in range(1, 7):
            j = i + d
            row.append(length if j >= length else links.get(j, j))
        dest[i] = row
    return dest[1:]


def expected_rolls(board: Dict, probs: Sequence[float]) -> float:
    """Expected number of throws starting from square 1."""
    n = board["length"]
    links = dict(board["links"])

    A = np.zeros((n, n))
    b = np.ones(n)
    A[-1, -1], b[-1] = 1, 0                        # absorbing state

    dests = _destinations(n, links)

    for i in range(n - 1):
        A[i, i] = 1
        for p, d in zip(probs, dests[i]):
            A[i, d - 1] -= p                      # transition weights

    return float(np.linalg.solve(A, b)[0])


# ────────────────────────────────────────────────────────────────────
# Optimiser (Point 4)
# ────────────────────────────────────────────────────────────────────
def optimise_die(board: Dict,
                 x0: Sequence[float] | None = None
                 ) -> Tuple[Tuple[float, ...], float]:
    """
    Parameters
    ----------
    board : dict
        Board description: {'length': int, 'links': [(start,end), …]}.
    x0    : optional initial guess (defaults to fair die).

    Returns
    -------
    probs_opt : tuple[float]   Optimal probabilities (p1…p6).
    exp_rolls : float         Expected rolls with that distribution.
    """
    if x0 is None:
        x0 = np.full(6, 1 / 6)

    # Equality constraints: probabilities sum to 1 and mean step = 3.5
    constraints = (
        {"type": "eq", "fun": lambda p: np.sum(p) - 1},
        {"type": "eq", "fun": lambda p: np.dot(np.arange(1, 7), p) - 3.5},
    )
    bounds = [(0, 1)] * 6

    res = minimize(lambda p: expected_rolls(board, p),
                   x0,
                   method="SLSQP",
                   bounds=bounds,
                   constraints=constraints,
                   options={"ftol": 1e-9})

    if not res.success:
        raise RuntimeError(f"Optimisation failed: {res.message}")

    p_opt = tuple(res.x)
    exp_opt = expected_rolls(board, p_opt)
    return p_opt, exp_opt


# ────────────────────────────────────────────────────────────────────
# Demo (executed when run as script)
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_board = {"length": 9, "links": [(2, 4), (7, 3), (6, 8)]}
    p, E = optimise_die(demo_board)
    print("Optimal probability vector:", [round(pi, 4) for pi in p])
    print(f"Expected rolls: {E:.6f}")
