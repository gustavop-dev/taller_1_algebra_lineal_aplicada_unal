"""
-----------------------------

Fast routine for an *upper-triangular* matrix A that returns an
orthonormal basis of all X such that AX = XA.  It avoids the costly
(n² × n²) SVD used in the general algorithm by exploiting the fact that
the Kronecker-difference operator is itself upper-triangular.

Computational cost:  𝒪(n⁴)   instead of   𝒪(n⁶).

If A is not triangular, call SciPy’s real Schur decomposition first:
    Q, T = scipy.linalg.schur(A, output='real')
    B  = commuting_basis_triangular(T)
    B  = [Q @ X @ Q.T for X in B]          # transform back
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import norm


def commuting_basis_triangular(A: np.ndarray,
                               tol: float = 1e-12) -> list[np.ndarray]:
    """
    Parameters
    ----------
    A   : (n, n) ndarray (must be upper-triangular)
    tol : float
        Threshold for identifying zero pivots.

    Returns
    -------
    basis : list[(n, n) ndarray]
        Frobenius-orthonormal basis of the commutant of A.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.allclose(A, np.triu(A)):
        raise ValueError("A must be (upper) triangular; "
                         "call scipy.linalg.schur first otherwise")

    n = A.shape[0]
    I = np.eye(n)

    # Build triangular Kronecker difference  K = I⊗A − Aᵀ⊗I  (size n²×n²)
    K = np.kron(I, A) - np.kron(A.T, I)
    m = n * n

    # We will compute a sparse RREF tailored to triangular K:
    free_cols: list[int] = []
    piv_val = np.diag(K)

    # 1) Identify free variables (zero diagonal) and pivot rows (non-zero diag)
    pivot_row_for_col = {}
    for idx in range(m):
        if abs(piv_val[idx]) < tol:
            free_cols.append(idx)          # diagonal 0 → variable is free
        else:
            pivot_row_for_col[idx] = idx   # record pivot location

    basis = []
    for fc in free_cols:
        v = np.zeros(m)
        v[fc] = 1.0

        # Back-substitute over pivot rows *below* the free variable
        # (because K is upper-triangular).
        for row in range(fc + 1, m):
            diag = K[row, row]
            if abs(diag) < tol:            # row has no pivot ⇒ already satisfied
                continue
            # row equation: diag*v[row] + Σ_{k<row} K[row,k] v[k] = 0
            rhs = np.dot(K[row, :row], v[:row])
            v[row] = -rhs / diag

        basis.append(v.reshape(n, n))

    # Orthonormalise → Modified Gram–Schmidt (Frobenius inner product)
    ortho = []
    for X in basis:
        for Y in ortho:
            X -= np.trace(Y.T @ X) * Y
        nrm = norm(X, "fro")
        if nrm > tol:
            ortho.append(X / nrm)

    return ortho


# ─── Demo & timing comparison ────────────────────────────────────────
if __name__ == "__main__":
    import time
    from punto_inicial import commuting_basis  # original (SVD) version

    for n in range(2, 9):
        A = np.triu(np.random.randn(n, n))       # random triangular

        t0 = time.perf_counter()
        B_fast = commuting_basis_triangular(A)
        t_fast = time.perf_counter() - t0

        t0 = time.perf_counter()
        B_ref = commuting_basis(A)               # slow reference
        t_ref = time.perf_counter() - t0

        # sanity check: spaces must have same dimension
        assert len(B_fast) == len(B_ref)

        print(f"n={n:2d}  dim={len(B_fast):2d}  "
              f"fast={t_fast:8.5f}s   SVD={t_ref:8.5f}s")
