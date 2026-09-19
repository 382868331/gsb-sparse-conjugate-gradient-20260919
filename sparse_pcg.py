"""Sparse SPD linear solver: COO -> CSR assembly and preconditioned CG.

Pure Python 3.14 standard library; no third-party dependencies.

Public API
----------
csr_from_coo(n, triplets) -> CSRMatrix
    Assemble a symmetric sparse matrix from COO triplets ``(i, j, v)``.
    Duplicates are merged with ``math.fsum``; exact zeros are dropped.
    Symmetry is verified exactly after merging (missing entries count as
    0.0); nothing is symmetrized or densified implicitly.

solve_pcg(A, b, x0=None, *, atol, rtol, max_iter, preconditioner) -> PCGResult
    Preconditioned conjugate gradient with Jacobi or IC(0) preconditioning.
    Convergence is always judged by the true residual ``||b - A x||_2``
    against ``atol + rtol * ||b||_2``; the recurrence residual is never
    reported as if it were the true one.

Positive-definiteness of ``A`` is a caller premise and is not fully
pre-checked; numerical failures (non-positive pivot, non-positive
curvature, non-finite intermediate values) are reported as
``status == "breakdown"`` instead of raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CSRMatrix",
    "PCGResult",
    "SparseMatrixError",
    "csr_from_coo",
    "solve_pcg",
]

_MAX_N = 256
_MAX_TRIPLETS = 10000
_MAX_ITER_LIMIT = 500
_PRECONDITIONERS = ("jacobi", "ic0")


class SparseMatrixError(ValueError):
    """Raised when COO input data or solver arguments are invalid."""


class _Breakdown(Exception):
    """Internal: numerical breakdown detected during PCG."""


class CSRMatrix:
    """Row-compressed sparse matrix (CSR), float64 values."""

    __slots__ = ("n", "indptr", "indices", "values")

    def __init__(self, n, indptr, indices, values):
        self.n = n
        self.indptr = indptr
        self.indices = indices
        self.values = values

    @property
    def nnz(self):
        return len(self.values)

    def matvec(self, x):
        """Return ``A @ x`` for a dense vector ``x`` of length ``n``."""
        if len(x) != self.n:
            raise ValueError(
                f"vector length {len(x)} does not match matrix order {self.n}"
            )
        indptr, indices, values = self.indptr, self.indices, self.values
        y = [0.0] * self.n
        for i in range(self.n):
            y[i] = math.fsum(
                values[p] * x[indices[p]] for p in range(indptr[i], indptr[i + 1])
            )
        return y

    def diagonal(self):
        """Return the diagonal as a dense list (missing entries are 0.0)."""
        indptr, indices, values = self.indptr, self.indices, self.values
        d = [0.0] * self.n
        for i in range(self.n):
            for p in range(indptr[i], indptr[i + 1]):
                if indices[p] == i:
                    d[i] = values[p]
                    break
        return d


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def csr_from_coo(n, triplets):
    """Assemble a symmetric CSR matrix from COO triplets ``(i, j, v)``.

    Rules (see module docstring): duplicates merged with ``math.fsum``,
    exact zeros dropped, entries sorted by (row, col), symmetry checked
    exactly after merging. Invalid indices or non-finite values are
    rejected with :class:`SparseMatrixError`.
    """
    if not _is_int(n) or not 1 <= n <= _MAX_N:
        raise SparseMatrixError(f"n must be an integer in [1, {_MAX_N}], got {n!r}")
    triplets = list(triplets)
    if len(triplets) > _MAX_TRIPLETS:
        raise SparseMatrixError(
            f"at most {_MAX_TRIPLETS} triplets allowed, got {len(triplets)}"
        )

    acc = {}
    for pos, t in enumerate(triplets):
        try:
            i, j, v = t
        except (TypeError, ValueError):
            raise SparseMatrixError(
                f"triplet #{pos} is not an (i, j, v) triple: {t!r}"
            ) from None
        if not _is_int(i) or not _is_int(j) or not (0 <= i < n) or not (0 <= j < n):
            raise SparseMatrixError(
                f"triplet #{pos} has out-of-range index: ({i!r}, {j!r}) for n={n}"
            )
        if not _is_real(v) or not math.isfinite(v):
            raise SparseMatrixError(
                f"triplet #{pos} has a non-finite or non-real value: {v!r}"
            )
        acc.setdefault((i, j), []).append(float(v))

    merged = {}
    for key, vals in acc.items():
        s = math.fsum(vals)
        if s != 0.0:  # drop exact zeros
            merged[key] = s

    # Exact symmetry check after merging; missing entries count as 0.0.
    for (i, j), v in merged.items():
        if i != j and merged.get((j, i), 0.0) != v:
            raise SparseMatrixError(
                f"matrix is not exactly symmetric: A[{i},{j}]={v!r} but "
                f"A[{j},{i}]={merged.get((j, i), 0.0)!r}"
            )

    entries = sorted(merged.items())
    indptr = [0] * (n + 1)
    indices = []
    values = []
    for (i, j), v in entries:
        indptr[i + 1] += 1
        indices.append(j)
        values.append(v)
    for i in range(n):
        indptr[i + 1] += indptr[i]
    return CSRMatrix(n, indptr, indices, values)


@dataclass
class PCGResult:
    """Outcome of :func:`solve_pcg`.

    ``status`` is one of ``"converged"``, ``"max_iter"``, ``"breakdown"``.
    ``residual_history`` is a list of ``(iteration, true_residual_norm)``
    pairs; every recorded norm is a recomputed true residual
    ``||b - A x||_2``, never the recurrence residual. ``matvecs`` is the
    actual number of matrix-vector products performed (including those
    used for true-residual recomputation).
    """

    x: list
    status: str
    iterations: int
    residual_history: list
    matvecs: int
    preconditioner: str
    message: str = ""

    @property
    def converged(self):
        return self.status == "converged"


def _norm(v):
    return math.sqrt(math.fsum(t * t for t in v))


def _dot(u, v):
    return math.fsum(a * b for a, b in zip(u, v))


class _Jacobi:
    def __init__(self, diag):
        self._inv = [1.0 / d for d in diag]

    def apply(self, r):
        return [ri * di for ri, di in zip(r, self._inv)]


def _build_jacobi(A):
    diag = A.diagonal()
    for i, d in enumerate(diag):
        if not math.isfinite(d) or d <= 0.0:
            raise _Breakdown(
                f"Jacobi preconditioner: non-positive diagonal at row {i}: {d!r}"
            )
    return _Jacobi(diag)


class _IC0:
    """IC(0): incomplete Cholesky on the exact lower-triangle pattern of A.

    No fill-in, no perturbation, no implicit fallback. A non-positive or
    non-finite pivot raises :class:`_Breakdown`.
    """

    def __init__(self, rows, cols):
        self._rows = rows  # rows[i]: dict j -> L[i, j] for j <= i
        self._cols = cols  # cols[j]: list of (i, L[i, j]) for i > j

    def apply(self, r):
        rows, cols = self._rows, self._cols
        n = len(rows)
        y = [0.0] * n
        for i in range(n):
            Li = rows[i]
            s = r[i] - math.fsum(v * y[j] for j, v in Li.items() if j < i)
            y[i] = s / Li[i]
        z = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i] - math.fsum(v * z[j] for j, v in cols[i])
            z[i] = s / rows[i][i]
        return z


def _build_ic0(A):
    n = A.n
    indptr, indices, values = A.indptr, A.indices, A.values
    lower = [dict() for _ in range(n)]  # strict lower part of A
    diag = [0.0] * n
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j < i:
                lower[i][j] = values[p]
            elif j == i:
                diag[i] = values[p]

    L = [dict() for _ in range(n)]  # L[i][j] for j <= i, includes diagonal
    for i in range(n):
        Li = L[i]
        for k in sorted(lower[i]):
            Lk = L[k]
            s = lower[i][k] - math.fsum(
                Li[j] * Lk[j] for j in Li.keys() & Lk.keys()
            )
            s /= Lk[k]
            if not math.isfinite(s):
                raise _Breakdown(
                    f"IC(0): non-finite off-diagonal value at ({i}, {k})"
                )
            Li[k] = s
        d = diag[i] - math.fsum(v * v for v in Li.values())
        if not math.isfinite(d) or d <= 0.0:
            raise _Breakdown(f"IC(0): non-positive pivot at row {i}: {d!r}")
        Li[i] = math.sqrt(d)

    cols = [[] for _ in range(n)]
    for i in range(n):
        for j, v in L[i].items():
            if j < i:
                cols[j].append((i, v))
    return _IC0(L, cols)


def _build_preconditioner(A, kind):
    if kind == "jacobi":
        return _build_jacobi(A)
    return _build_ic0(A)


def _validate_tolerances(atol, rtol):
    for name, t in (("atol", atol), ("rtol", rtol)):
        if not _is_real(t) or not math.isfinite(t) or t < 0.0:
            raise SparseMatrixError(
                f"{name} must be a finite non-negative real, got {t!r}"
            )
    if atol == 0.0 and rtol == 0.0:
        raise SparseMatrixError("at least one of atol/rtol must be positive")


def _validate_vector(name, v, n):
    v = list(v)
    if len(v) != n:
        raise SparseMatrixError(f"{name} must have length {n}, got {len(v)}")
    out = []
    for k, t in enumerate(v):
        if not _is_real(t) or not math.isfinite(t):
            raise SparseMatrixError(f"{name}[{k}] is not a finite real: {t!r}")
        out.append(float(t))
    return out


def solve_pcg(A, b, x0=None, *, atol=1e-10, rtol=1e-8, max_iter=500,
              preconditioner="jacobi"):
    """Solve ``A x = b`` for a symmetric positive-definite CSR matrix ``A``.

    Parameters
    ----------
    A : CSRMatrix
        Symmetric matrix from :func:`csr_from_coo`. Positive-definiteness
        is a caller premise.
    b : sequence of float
        Right-hand side, length ``A.n``, finite values.
    x0 : sequence of float, optional
        Initial guess (default: zero vector).
    atol, rtol : float
        Finite, non-negative, at least one positive. Convergence requires
        the *true* residual ``||b - A x||_2 <= atol + rtol * ||b||_2``.
    max_iter : int
        Iteration budget, 0 <= max_iter <= 500.
    preconditioner : {"jacobi", "ic0"}
        Jacobi (diagonal) or IC(0) (pattern-preserving incomplete
        Cholesky, no fill-in, no perturbation).

    Returns
    -------
    PCGResult
        ``status`` is "converged", "max_iter" or "breakdown". The true
        residual is recomputed every 10 iterations, at every candidate
        convergence, and at exit; if a candidate fails the true-residual
        test, the search direction is restarted from the true residual.
    """
    if not isinstance(A, CSRMatrix):
        raise SparseMatrixError(f"A must be a CSRMatrix, got {type(A).__name__}")
    n = A.n
    b = _validate_vector("b", b, n)
    x = [0.0] * n if x0 is None else _validate_vector("x0", x0, n)
    _validate_tolerances(atol, rtol)
    if not _is_int(max_iter) or not 0 <= max_iter <= _MAX_ITER_LIMIT:
        raise SparseMatrixError(
            f"max_iter must be an integer in [0, {_MAX_ITER_LIMIT}], got {max_iter!r}"
        )
    if preconditioner not in _PRECONDITIONERS:
        raise SparseMatrixError(
            f"preconditioner must be one of {_PRECONDITIONERS}, got {preconditioner!r}"
        )

    matvecs = 0

    def mv(vec):
        nonlocal matvecs
        matvecs += 1
        return A.matvec(vec)

    def true_residual():
        ax = mv(x)
        return [bi - ai for bi, ai in zip(b, ax)]

    threshold = atol + rtol * _norm(b)
    history = []

    def record(it, rnorm):
        if history and history[-1][0] == it:
            history[-1] = (it, rnorm)
        else:
            history.append((it, rnorm))

    def result(status, it, message=""):
        return PCGResult(list(x), status, it, history, matvecs, preconditioner,
                         message)

    # True residual first: only build the preconditioner if the initial
    # guess does not already satisfy the tolerance.
    r = true_residual()
    rnorm = _norm(r)
    record(0, rnorm)
    if rnorm <= threshold:
        return result("converged", 0,
                      "initial guess already satisfies the tolerance")

    try:
        M = _build_preconditioner(A, preconditioner)
    except _Breakdown as exc:
        return result("breakdown", 0, str(exc))

    z = M.apply(r)
    rz = _dot(r, z)
    if not math.isfinite(rz) or rz <= 0.0:
        return result("breakdown", 0,
                      f"non-positive or non-finite preconditioner curvature "
                      f"r^T z = {rz!r}")
    p = list(z)

    it = 0
    while it < max_iter:
        ap = mv(p)
        pap = _dot(p, ap)
        if not math.isfinite(pap) or pap <= 0.0:
            return result("breakdown", it,
                          f"non-positive or non-finite curvature "
                          f"p^T A p = {pap!r} at iteration {it}")
        alpha = rz / pap
        if not math.isfinite(alpha):
            return result("breakdown", it,
                          f"non-finite step length at iteration {it}")
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
        it += 1
        rnorm = _norm(r)
        if not math.isfinite(rnorm):
            return result("breakdown", it,
                          f"non-finite residual at iteration {it}")

        if it % 10 == 0 or rnorm <= threshold:
            # Scheduled refresh or candidate convergence: recompute the
            # true residual and judge convergence by it alone.
            candidate = rnorm <= threshold
            r = true_residual()
            rnorm = _norm(r)
            record(it, rnorm)
            if rnorm <= threshold:
                return result("converged", it)
            if candidate:
                # Candidate rejected by the true residual: restart the
                # search direction from the true residual.
                z = M.apply(r)
                rz = _dot(r, z)
                if not math.isfinite(rz) or rz <= 0.0:
                    return result("breakdown", it,
                                  f"non-positive or non-finite preconditioner "
                                  f"curvature r^T z = {rz!r} at iteration {it}")
                p = list(z)
                continue
            # Scheduled refresh only: keep the conjugate direction and
            # continue with the beta update below using the refreshed r.

        z = M.apply(r)
        rz_new = _dot(r, z)
        if not math.isfinite(rz_new) or rz_new <= 0.0:
            return result("breakdown", it,
                          f"non-positive or non-finite preconditioner "
                          f"curvature r^T z = {rz_new!r} at iteration {it}")
        beta = rz_new / rz
        if not math.isfinite(beta):
            return result("breakdown", it,
                          f"non-finite beta at iteration {it}")
        for i in range(n):
            p[i] = z[i] + beta * p[i]
        rz = rz_new

    # Budget exhausted: recompute the true residual at exit before
    # reporting, in case the iterate actually meets the tolerance.
    r = true_residual()
    rnorm = _norm(r)
    record(it, rnorm)
    if rnorm <= threshold:
        return result("converged", it)
    return result("max_iter", it,
                  f"iteration budget {max_iter} exhausted; "
                  f"true residual {rnorm!r} > threshold {threshold!r}")
