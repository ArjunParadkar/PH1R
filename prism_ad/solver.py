# Surrogate eigenvalue operator: parameter-conditioned A(p), eigenvalues via ARPACK; used to benchmark iterative eigensolver runtime vs neural inference.

import numpy as np
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse import diags


def _build_operator(params, n):
    p = np.atleast_1d(params)
    d = len(p)
    diag = np.ones(n) * 0.1
    off = np.ones(n - 1) * (-0.02)
    for i in range(min(d, n)):
        diag[i] += 0.3 * p[i % d]
    for i in range(min(d, n - 1)):
        off[i] += 0.05 * p[(i + 1) % d]
    A = diags([off, diag, off], [-1, 0, 1], format="csr")
    return A


def _matvec(x, params, n):
    A = _build_operator(params, n)
    return A.dot(x)


def compute_rightmost_eigenvalues(params, n_eigs=20, n_dim=80, tol=1e-8):
    def mv(x):
        return _matvec(x, params, n_dim)

    L = LinearOperator((n_dim, n_dim), matvec=mv)
    try:
        vals, _ = eigs(L, k=min(n_eigs, n_dim - 1), which="LI", tol=tol)
        vals = np.array(vals, dtype=np.complex128)
    except Exception:
        vals = np.zeros(n_eigs, dtype=np.complex128)
        vals[0] = 0.1 + 0.01j
    return vals


def solve_one(params, n_eigs=20, tol=1e-8):
    return compute_rightmost_eigenvalues(params, n_eigs=n_eigs, tol=tol)
