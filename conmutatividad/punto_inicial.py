"""
------------------

Compute an orthonormal basis (Frobenius inner product) for the space
    C(A) = { X ∈ ℝ^{n×n} :  X A = A X }
of matrices that commute with a given square matrix A.

Method
------
Write the commutation constraint in vectorised (Kronecker) form

    (I ⊗ A - Aᵀ ⊗ I) vec(X) = 0 .

The null-space of the coefficient matrix gives all solutions vec(X).
A singular–value decomposition yields an orthonormal basis of that
null-space; reshaping each basis vector back to n×n provides the
desired matrices.

The resulting list is orthonormal under the Frobenius inner product

    ⟨X, Y⟩ = trace(Xᵀ Y) = vec(X)ᵀ vec(Y).
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import svd


def commuting_basis(A: np.ndarray, tol: float = 1e-10) -> list[np.ndarray]:
    """
    Parameters
    ----------
    A   : (n, n) ndarray
        Square input matrix.
    tol : float, optional
        Tolerance to decide which singular values are zero.

    Returns
    -------
    basis : list of (n, n) ndarrays
        Orthonormal basis (Frobenius‐orthogonal) for the commutant of A.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    I = np.eye(n)

    # Build the (n² × n²) commutation matrix  K = I⊗A − Aᵀ⊗I
    K = np.kron(I, A) - np.kron(A.T, I)

    # SVD to extract null-space
    U, s, Vh = svd(K, full_matrices=True)
    null_mask = s < tol
    if not np.any(null_mask):              # A commutes only with scalar multiples of I
        return [I / np.sqrt(n)]            # normalised identity matrix

    # Columns of V corresponding to zero singular values
    null_vectors = Vh.T[:, null_mask]

    # Orthonormalise (already orthonormal from SVD) and reshape
    basis = [v.reshape(n, n) for v in null_vectors.T]

    return basis


# ── Demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: A is a 3×3 diagonal matrix with repeated eigenvalues
    A = np.diag([1, 2, 2])

    B = commuting_basis(A)
    print(f"Dimension of commutant: {len(B)}")
    for i, X in enumerate(B, 1):
        print(f"Basis matrix {i}:\n{X}\n")

    # Verify orthonormality and commutation
    for i, Xi in enumerate(B):
        for j, Xj in enumerate(B):
            prod = np.trace(Xi.T @ Xj)
            if i == j:
                assert np.isclose(prod, 1), "basis not normalised"
            else:
                assert np.isclose(prod, 0), "basis not orthogonal"
            assert np.allclose(Xi @ A, A @ Xi), "does not commute"
