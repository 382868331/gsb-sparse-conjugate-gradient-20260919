"""Tests for sparse_pcg: COO->CSR assembly, PCG semantics, failure paths.

Validation strategy (per TASK.md): an independent dense Gaussian
elimination reference for n <= 8, a sparse case with a known solution,
and deliberately triggered failure cases. Random checks use a fixed
seed and a few small samples only.
"""

import math
import random
import unittest

from sparse_pcg import (
    PCGResult,
    SparseMatrixError,
    csr_from_coo,
    solve_pcg,
)


# ---------------------------------------------------------------------------
# Independent reference: dense Gaussian elimination with partial pivoting.
# Only used for n <= 8 in these tests.
# ---------------------------------------------------------------------------

def dense_from_csr(A):
    M = [[0.0] * A.n for _ in range(A.n)]
    for i in range(A.n):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            M[i][A.indices[p]] = A.values[p]
    return M


def gaussian_solve(M, b):
    n = len(M)
    a = [list(M[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0.0:
            raise ValueError("singular matrix")
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = a[i][n] - math.fsum(a[i][c] * x[c] for c in range(i + 1, n))
        x[i] = s / a[i][i]
    return x


def poisson_1d(n):
    """Symmetric positive-definite 1D Poisson matrix as COO triplets."""
    t = []
    for i in range(n):
        t.append((i, i, 2.0))
        if i > 0:
            t.append((i, i - 1, -1.0))
            t.append((i - 1, i, -1.0))
    return t


def true_residual_norm(A, x, b):
    ax = A.matvec(x)
    return math.sqrt(math.fsum((bi - ai) ** 2 for bi, ai in zip(b, ax)))


class TestCooAssembly(unittest.TestCase):
    def test_duplicate_coo_merged_with_fsum(self):
        # (0,1) and (1,0) each appear twice and must merge to equal sums.
        A = csr_from_coo(2, [
            (0, 0, 2.0),
            (0, 1, 1.0), (0, 1, 0.5),
            (1, 0, 1.25), (1, 0, 0.25),
            (1, 1, 4.0),
        ])
        self.assertEqual(A.nnz, 4)
        M = dense_from_csr(A)
        self.assertEqual(M[0][1], 1.5)
        self.assertEqual(M[1][0], 1.5)

    def test_fsum_accuracy_and_exact_zero_dropped(self):
        # Naive left-to-right summation of [1e16, 1, -1e16] gives 0.0;
        # math.fsum gives exactly 1.0. The (0,1)/(1,0) pairs cancel to an
        # exact zero and must be removed entirely.
        A = csr_from_coo(2, [
            (0, 0, 1e16), (0, 0, 1.0), (0, 0, -1e16),
            (0, 1, 1.0), (0, 1, -1.0),
            (1, 0, 0.5), (1, 0, -0.5),
            (1, 1, 3.0),
        ])
        self.assertEqual(A.nnz, 2)  # only the two diagonal entries remain
        M = dense_from_csr(A)
        self.assertEqual(M[0][0], 1.0)
        self.assertEqual(M[1][1], 3.0)
        self.assertEqual(M[0][1], 0.0)
        self.assertEqual(M[1][0], 0.0)

    def test_sorted_canonical_order(self):
        A = csr_from_coo(3, [
            (2, 2, 5.0), (0, 2, 1.0), (2, 0, 1.0),
            (1, 1, 4.0), (0, 0, 3.0),
        ])
        order = []
        for i in range(3):
            for p in range(A.indptr[i], A.indptr[i + 1]):
                order.append((i, A.indices[p]))
        self.assertEqual(order, sorted(order))

    def test_asymmetry_rejected_value_mismatch(self):
        with self.assertRaises(SparseMatrixError):
            csr_from_coo(2, [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0),
                             (1, 1, 1.0)])

    def test_asymmetry_rejected_missing_counterpart(self):
        # Missing (1,0) counts as 0.0, which differs from A[0,1] = 1.0.
        with self.assertRaises(SparseMatrixError):
            csr_from_coo(2, [(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)])

    def test_invalid_inputs_rejected(self):
        bad = [
            [(0, 0, 1.0), (0, 2, 1.0)],          # index out of range
            [(0, 0, 1.0), (-1, 0, 1.0)],         # negative index
            [(0, 0, float("nan"))],              # NaN value
            [(0, 0, float("inf"))],              # inf value
            [(0, 0.5, 1.0)],                     # non-integer index
            [(0, True, 1.0)],                    # bool is not an index
            [(0, 0)],                            # not a triple
        ]
        for triplets in bad:
            with self.assertRaises(SparseMatrixError, msg=str(triplets)):
                csr_from_coo(2, triplets)
        with self.assertRaises(SparseMatrixError):
            csr_from_coo(0, [])                  # n too small
        with self.assertRaises(SparseMatrixError):
            csr_from_coo(257, [])                # n too large
        with self.assertRaises(SparseMatrixError):
            csr_from_coo(2.0, [])                # n not an int
        with self.assertRaises(SparseMatrixError):
            csr_from_coo(1, [(0, 0, 1.0)] * 10001)  # too many triplets


class TestArgumentValidation(unittest.TestCase):
    def setUp(self):
        self.A = csr_from_coo(2, [(0, 0, 2.0), (1, 1, 3.0)])

    def test_tolerance_rules(self):
        for atol, rtol in [(0.0, 0.0), (-1.0, 1.0), (1.0, -1.0),
                           (float("nan"), 1.0), (1.0, float("inf"))]:
            with self.assertRaises(SparseMatrixError, msg=f"{atol},{rtol}"):
                solve_pcg(self.A, [1.0, 1.0], atol=atol, rtol=rtol)

    def test_max_iter_bounds(self):
        for mi in (-1, 501, 1.5):
            with self.assertRaises(SparseMatrixError, msg=str(mi)):
                solve_pcg(self.A, [1.0, 1.0], max_iter=mi)

    def test_vector_validation(self):
        with self.assertRaises(SparseMatrixError):
            solve_pcg(self.A, [1.0])                      # wrong length
        with self.assertRaises(SparseMatrixError):
            solve_pcg(self.A, [1.0, float("nan")])        # non-finite b
        with self.assertRaises(SparseMatrixError):
            solve_pcg(self.A, [1.0, 1.0], x0=[0.0])       # wrong x0 length

    def test_unknown_preconditioner(self):
        with self.assertRaises(SparseMatrixError):
            solve_pcg(self.A, [1.0, 1.0], preconditioner="ssor")


class TestSolveSemantics(unittest.TestCase):
    def test_scaled_diagonal_jacobi_one_iteration(self):
        # Widely scaled diagonal: Jacobi preconditioner equals A^-1, so
        # PCG must converge in exactly one iteration and stay accurate.
        d = [1e8, 1.0, 1e-8, 42.0]
        A = csr_from_coo(4, [(i, i, v) for i, v in enumerate(d)])
        x_true = [1.0, -2.0, 3.0, 0.5]
        b = [di * xi for di, xi in zip(d, x_true)]
        res = solve_pcg(A, b, atol=1e-12, rtol=1e-12,
                        preconditioner="jacobi")
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.iterations, 1)
        for got, want in zip(res.x, x_true):
            self.assertAlmostEqual(got, want, places=12)

    def test_zero_rhs_nonzero_initial_guess(self):
        # b = 0 with x0 != 0: the initial true residual is -A x0 != 0,
        # and the solver must drive x toward 0.
        n = 10
        A = csr_from_coo(n, poisson_1d(n))
        b = [0.0] * n
        x0 = [1.0] * n
        res = solve_pcg(A, b, x0=x0, atol=1e-12, rtol=0.0,
                        preconditioner="ic0")
        self.assertEqual(res.status, "converged")
        self.assertGreater(res.iterations, 0)
        self.assertLess(math.sqrt(math.fsum(v * v for v in res.x)), 1e-8)
        # Threshold here is atol + rtol*||b|| = atol (||b|| = 0).
        self.assertLessEqual(res.residual_history[-1][1], 1e-12)

    def test_zero_iteration_when_initial_guess_solves(self):
        d = [2.0, 3.0, 4.0]
        A = csr_from_coo(3, [(i, i, v) for i, v in enumerate(d)])
        x_true = [1.5, -2.0, 0.25]
        b = [di * xi for di, xi in zip(d, x_true)]
        res = solve_pcg(A, b, x0=x_true, preconditioner="ic0")
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.iterations, 0)
        self.assertEqual(len(res.residual_history), 1)
        it, rnorm = res.residual_history[0]
        self.assertEqual(it, 0)
        self.assertLessEqual(rnorm, 1e-10 + 1e-8 * math.sqrt(
            math.fsum(v * v for v in b)))
        self.assertEqual(res.matvecs, 1)  # only the initial true residual

    def test_max_iter_exhausted(self):
        n = 50
        A = csr_from_coo(n, poisson_1d(n))
        b = [1.0] * n
        res = solve_pcg(A, b, max_iter=3, preconditioner="jacobi")
        self.assertEqual(res.status, "max_iter")
        self.assertEqual(res.iterations, 3)
        self.assertGreater(res.residual_history[-1][1], 0.0)

    def test_max_iter_zero(self):
        A = csr_from_coo(2, [(0, 0, 2.0), (1, 1, 2.0)])
        res = solve_pcg(A, [1.0, 1.0], max_iter=0)
        self.assertEqual(res.status, "max_iter")
        self.assertEqual(res.iterations, 0)

    def test_true_residual_reported_not_recurrence(self):
        n = 30
        A = csr_from_coo(n, poisson_1d(n))
        x_true = [math.sin(i) for i in range(n)]
        b = A.matvec(x_true)
        res = solve_pcg(A, b, atol=1e-11, rtol=1e-11, preconditioner="ic0")
        self.assertEqual(res.status, "converged")
        # The last recorded residual must be the true residual of the
        # returned vector, recomputed independently here.
        final_true = true_residual_norm(A, res.x, b)
        self.assertAlmostEqual(res.residual_history[-1][1], final_true,
                               places=15)
        self.assertLessEqual(final_true,
                             1e-11 + 1e-11 * math.sqrt(
                                 math.fsum(v * v for v in b)))
        # History: iteration numbers start at 0 and are non-decreasing;
        # every recorded value is a finite true residual.
        its = [it for it, _ in res.residual_history]
        self.assertEqual(its[0], 0)
        self.assertEqual(its, sorted(its))
        for _, rn in res.residual_history:
            self.assertTrue(math.isfinite(rn))
        # Matvec accounting: at least one per iteration plus the initial
        # and final true-residual recomputations.
        self.assertGreaterEqual(res.matvecs, res.iterations + 1)

    def test_residual_refresh_every_10_iterations(self):
        # A problem needing > 10 iterations must record a true residual
        # at iteration 10 (the scheduled refresh).
        n = 60
        A = csr_from_coo(n, poisson_1d(n))
        b = [1.0] * n
        res = solve_pcg(A, b, atol=1e-12, rtol=1e-12,
                        preconditioner="jacobi")
        self.assertGreaterEqual(res.iterations, 10)
        its = {it for it, _ in res.residual_history}
        self.assertIn(10, its)


class TestBreakdown(unittest.TestCase):
    def test_ic0_pivot_failure(self):
        # Symmetric but indefinite: IC(0) pivot at row 1 is 1 - 4 = -3.
        A = csr_from_coo(2, [(0, 0, 1.0), (0, 1, 2.0),
                             (1, 0, 2.0), (1, 1, 1.0)])
        res = solve_pcg(A, [1.0, 1.0], preconditioner="ic0")
        self.assertEqual(res.status, "breakdown")
        self.assertIn("pivot", res.message)
        self.assertEqual(res.iterations, 0)

    def test_jacobi_nonpositive_diagonal(self):
        # Zero diagonal entry (missing diagonal counts as 0.0).
        A = csr_from_coo(2, [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)])
        res = solve_pcg(A, [1.0, 1.0], preconditioner="jacobi")
        self.assertEqual(res.status, "breakdown")
        self.assertIn("diagonal", res.message)

    def test_breakdown_result_shape(self):
        A = csr_from_coo(2, [(0, 0, 1.0), (0, 1, 2.0),
                             (1, 0, 2.0), (1, 1, 1.0)])
        res = solve_pcg(A, [1.0, 1.0], preconditioner="ic0")
        self.assertIsInstance(res, PCGResult)
        self.assertFalse(res.converged)
        self.assertEqual(res.residual_history[0][0], 0)
        self.assertGreaterEqual(res.matvecs, 1)


class TestAgainstReference(unittest.TestCase):
    def test_known_solution_sparse_case(self):
        # Sparse SPD case with a known solution, both preconditioners.
        n = 40
        A = csr_from_coo(n, poisson_1d(n))
        x_true = [math.cos(0.3 * i) for i in range(n)]
        b = A.matvec(x_true)
        for pre in ("jacobi", "ic0"):
            res = solve_pcg(A, b, atol=1e-10, rtol=1e-10,
                            preconditioner=pre)
            self.assertEqual(res.status, "converged", msg=pre)
            err = max(abs(g - t) for g, t in zip(res.x, x_true))
            self.assertLess(err, 1e-6, msg=pre)

    def test_random_small_spd_vs_gaussian_reference(self):
        # Fixed seed, a few small samples (n <= 8): compare PCG against
        # the independent Gaussian elimination reference.
        rng = random.Random(20260920)
        for case in range(6):
            n = rng.randint(2, 8)
            trips = {}
            for i in range(n):
                rowsum = 0.0
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        v = rng.uniform(-0.4, 0.4)
                        trips[(i, j)] = v
                        trips[(j, i)] = v
                        rowsum += abs(v)
                # Diagonally dominant => SPD.
                off_rowsum = sum(abs(v) for (a, b), v in trips.items()
                                 if a == i or b == i)
                trips[(i, i)] = off_rowsum + 1.0
            A = csr_from_coo(n, sorted((i, j, v)
                                       for (i, j), v in trips.items()))
            b = [rng.uniform(-1.0, 1.0) for _ in range(n)]
            x_ref = gaussian_solve(dense_from_csr(A), b)
            for pre in ("jacobi", "ic0"):
                res = solve_pcg(A, b, atol=1e-12, rtol=1e-12,
                                preconditioner=pre)
                self.assertEqual(res.status, "converged",
                                 msg=f"case {case}, {pre}")
                err = max(abs(g - t) for g, t in zip(res.x, x_ref))
                self.assertLess(err, 1e-6, msg=f"case {case}, {pre}")

    def test_gaussian_reference_selfcheck(self):
        # Sanity: the reference solver itself solves a known 3x3 system.
        M = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        x_true = [1.0, 2.0, 3.0]
        b = [math.fsum(M[i][j] * x_true[j] for j in range(3))
             for i in range(3)]
        x = gaussian_solve(M, b)
        for got, want in zip(x, x_true):
            self.assertAlmostEqual(got, want, places=12)


if __name__ == "__main__":
    unittest.main()
