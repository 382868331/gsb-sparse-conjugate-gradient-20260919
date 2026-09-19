"""sparse_pcg 的单元测试。

覆盖:重复 COO 合并、非对称拒绝、缩放对角、IC 主元失败、零右端非零初值、
0 迭代初值解、max_iter、真实残差记录,以及 n<=8 高斯消元参考对照和
已知解稀疏案例。随机用例使用固定种子。
"""

import math
import random
import unittest

from sparse_pcg import (
    STATUS_BREAKDOWN,
    STATUS_CONVERGED,
    STATUS_MAX_ITER,
    csr_from_coo,
    pcg_solve,
)


# ---------------------------------------------------------------- 辅助工具

def dense_from_csr(A):
    n = A.n
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            M[i][A.indices[k]] = A.data[k]
    return M


def dense_matvec(M, x):
    return [math.fsum(M[i][j] * x[j] for j in range(len(x))) for i in range(len(M))]


def true_residual_norm(M, b, x):
    r = [bi - ai for bi, ai in zip(b, dense_matvec(M, x))]
    return math.sqrt(math.fsum(v * v for v in r))


def gauss_solve(M, b):
    """独立参考:部分主元高斯消元(仅用于 n<=8 的小系统)。"""
    n = len(M)
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) == 0.0:
            raise AssertionError("reference solver: singular matrix")
        A[col], A[piv] = A[piv], A[col]
        for r in range(col + 1, n):
            f = A[r][col] / A[col][col]
            for c in range(col, n + 1):
                A[r][c] -= f * A[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (A[i][n] - math.fsum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def triplets_from_dense(M):
    return [(i, j, M[i][j]) for i in range(len(M)) for j in range(len(M)) if M[i][j] != 0.0]


def random_spd_dense(rng, n):
    """A = L L^T + n*I,L 为随机下三角,保证 SPD 且对角占优。"""
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            L[i][j] = rng.uniform(-1.0, 1.0)
    A = [[math.fsum(L[i][k] * L[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] += n
    return A


def poisson_1d(n):
    """一维 Poisson(三对角 2,-1),返回三元组。"""
    t = []
    for i in range(n):
        t.append((i, i, 2.0))
        if i > 0:
            t.append((i, i - 1, -1.0))
        if i < n - 1:
            t.append((i, i + 1, -1.0))
    return t


# ---------------------------------------------------------------- COO -> CSR

class TestCsrFromCoo(unittest.TestCase):
    def test_duplicate_entries_merged_with_fsum(self):
        # 0.1 + 0.2 的 fsum 是正确舍入结果,与普通加法不同路径
        A = csr_from_coo(2, [(0, 0, 0.1), (0, 0, 0.2), (1, 1, 4.0)])
        self.assertEqual(A.data[0], math.fsum([0.1, 0.2]))
        self.assertEqual(A.data[0], 0.30000000000000004)

    def test_exact_zero_entries_removed(self):
        A = csr_from_coo(2, [(0, 0, 1.0), (0, 0, -1.0), (1, 1, 2.0), (0, 1, 0.0)])
        self.assertEqual(A.nnz, 1)
        self.assertEqual(dense_from_csr(A), [[0.0, 0.0], [0.0, 2.0]])

    def test_sorted_canonical_order(self):
        A = csr_from_coo(3, [(2, 0, 5.0), (0, 2, 5.0), (0, 0, 3.0), (2, 2, 4.0)])
        self.assertEqual(A.indices, [0, 2, 0, 2])
        self.assertEqual(A.indptr, [0, 2, 2, 4])

    def test_asymmetry_missing_counterpart_rejected(self):
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 0.5)])

    def test_asymmetry_unequal_counterpart_rejected(self):
        # 即使只差一个 ulp 也拒绝(要求精确相等)
        almost = math.nextafter(0.5, 1.0)
        self.assertNotEqual(almost, 0.5)
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 0.5), (1, 0, almost)])

    def test_exact_symmetry_accepted(self):
        A = csr_from_coo(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 0.5), (1, 0, 0.5)])
        self.assertEqual(A.nnz, 4)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            csr_from_coo(0, [])
        with self.assertRaises(ValueError):
            csr_from_coo(257, [])
        with self.assertRaises(ValueError):
            csr_from_coo(2.5, [])
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 2, 1.0)])          # 索引越界
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(-1, 0, 1.0)])         # 负索引
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 0, float("nan"))])  # 非有限值
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 0, float("inf"))])
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 0, "1.0")])         # 非数值
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(True, 0, 1.0)])        # 布尔不是合法索引
        with self.assertRaises(ValueError):
            csr_from_coo(2, [(0, 0, 1.0)] * 10001)   # 三元组超限


# ---------------------------------------------------------------- PCG 求解

class TestPcgSolve(unittest.TestCase):
    def test_scaled_diagonal_jacobi_converges_in_one_iteration(self):
        # 对角矩阵 + Jacobi:预条件器即 A 本身,一步收敛
        A = csr_from_coo(3, [(0, 0, 2.0), (1, 1, 8.0), (2, 2, 32.0)])
        b = [2.0, 16.0, -64.0]
        res = pcg_solve(A, b, atol=1e-12, rtol=0.0, preconditioner="jacobi")
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertEqual(res.iterations, 1)
        for xi, ei in zip(res.x, [1.0, 2.0, -2.0]):
            self.assertAlmostEqual(xi, ei, places=12)

    def test_ic0_pivot_failure_returns_breakdown(self):
        # 对称但非正定:[[1,2],[2,1]],IC(0) 第二个主元为 1-4=-3 <= 0
        A = csr_from_coo(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 2.0), (1, 0, 2.0)])
        res = pcg_solve(A, [1.0, 1.0], preconditioner="ic0")
        self.assertEqual(res.status, STATUS_BREAKDOWN)
        self.assertIn("pivot", res.message)

    def test_zero_rhs_nonzero_x0(self):
        # b = 0,非零初值:解应收敛到 0;容差为 atol(rtol*||b||=0)
        A = csr_from_coo(2, [(0, 0, 4.0), (1, 1, 4.0), (0, 1, 1.0), (1, 0, 1.0)])
        res = pcg_solve(A, [0.0, 0.0], x0=[3.0, -2.0], atol=1e-12, rtol=1e-8)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertLessEqual(math.hypot(*res.x), 1e-9)

    def test_zero_iteration_when_x0_already_solves(self):
        A = csr_from_coo(2, [(0, 0, 2.0), (1, 1, 4.0)])
        b = [2.0, 8.0]
        res = pcg_solve(A, b, x0=[1.0, 2.0], atol=1e-12, rtol=0.0)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.matvec_count, 1)  # 只算了初始真实残差
        self.assertEqual(res.residual_history[0][0], 0)
        self.assertEqual(res.x, [1.0, 2.0])

    def test_max_iter_zero_budget(self):
        A = csr_from_coo(2, [(0, 0, 2.0), (1, 1, 4.0)])
        res = pcg_solve(A, [1.0, 1.0], max_iter=0)
        self.assertEqual(res.status, STATUS_MAX_ITER)
        self.assertEqual(res.iterations, 0)

    def test_max_iter_exhausted(self):
        # 非对角 SPD,Jacobi 一步收敛不了;预算 1 应返回 max_iter
        A = csr_from_coo(3, [(0, 0, 4.0), (1, 1, 4.0), (2, 2, 4.0),
                             (0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0)])
        res = pcg_solve(A, [1.0, 2.0, 3.0], max_iter=1, atol=1e-14, rtol=0.0)
        self.assertEqual(res.status, STATUS_MAX_ITER)
        self.assertEqual(res.iterations, 1)
        # 退出时重算了真实残差并记录
        self.assertEqual(res.residual_history[-1][0], 1)
        M = dense_from_csr(A)
        self.assertAlmostEqual(res.residual_history[-1][1],
                               true_residual_norm(M, [1.0, 2.0, 3.0], res.x), places=15)

    def test_true_residual_history_is_honest(self):
        # 记录中的每个残差都应是真实残差:末条与独立重算一致,且收敛时达标
        n = 12
        A = csr_from_coo(n, poisson_1d(n))
        M = dense_from_csr(A)
        b = [float(i % 3 - 1) for i in range(n)]
        for pre in ("jacobi", "ic0"):
            res = pcg_solve(A, b, atol=1e-10, rtol=1e-10, max_iter=200, preconditioner=pre)
            self.assertEqual(res.status, STATUS_CONVERGED)
            iters = [it for it, _ in res.residual_history]
            self.assertEqual(iters[0], 0)
            self.assertEqual(iters, sorted(iters))
            self.assertEqual(iters[-1], res.iterations)
            final_true = true_residual_norm(M, b, res.x)
            self.assertAlmostEqual(res.residual_history[-1][1], final_true, places=12)
            tol = 1e-10 + 1e-10 * math.sqrt(math.fsum(v * v for v in b))
            self.assertLessEqual(final_true, tol)
            # 中间记录点也应是真实残差量级(与末条同数量级或更大)
            for _, rv in res.residual_history[:-1]:
                self.assertGreaterEqual(rv, final_true * 1e-3)
            self.assertGreaterEqual(res.matvec_count, res.iterations + 1)

    def test_residual_refresh_every_10_iterations(self):
        # Poisson + Jacobi 需要远超 10 次迭代,验证历史中出现第 10 次迭代的真实残差
        n = 40
        A = csr_from_coo(n, poisson_1d(n))
        b = [1.0] * n
        res = pcg_solve(A, b, atol=1e-10, rtol=1e-10, max_iter=500)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertGreater(res.iterations, 10)
        iters = [it for it, _ in res.residual_history]
        self.assertIn(10, iters)

    def test_random_small_spd_matches_gauss_reference(self):
        rng = random.Random(20260920)
        for trial in range(6):
            n = rng.randint(2, 8)
            M = random_spd_dense(rng, n)
            A = csr_from_coo(n, triplets_from_dense(M))
            b = [rng.uniform(-3, 3) for _ in range(n)]
            x_ref = gauss_solve(M, b)
            for pre in ("jacobi", "ic0"):
                res = pcg_solve(A, b, atol=1e-11, rtol=1e-11, max_iter=500, preconditioner=pre)
                self.assertEqual(res.status, STATUS_CONVERGED,
                                 msg=f"trial={trial} n={n} pre={pre}: {res.message}")
                for xi, ri in zip(res.x, x_ref):
                    self.assertAlmostEqual(xi, ri, delta=1e-6,
                                           msg=f"trial={trial} n={n} pre={pre}")

    def test_known_solution_sparse_case(self):
        # 已知解的稀疏案例:Poisson n=50,x_true 已知,b = A x_true
        n = 50
        A = csr_from_coo(n, poisson_1d(n))
        M = dense_from_csr(A)
        x_true = [math.sin(1.5 * i) + 0.25 * i for i in range(n)]
        b = dense_matvec(M, x_true)
        for pre in ("jacobi", "ic0"):
            res = pcg_solve(A, b, atol=1e-10, rtol=1e-10, max_iter=500, preconditioner=pre)
            self.assertEqual(res.status, STATUS_CONVERGED, msg=f"{pre}: {res.message}")
            err = max(abs(xi - ti) for xi, ti in zip(res.x, x_true))
            self.assertLess(err, 1e-6, msg=f"{pre}: max error {err}")

    def test_ic0_not_guaranteed_but_works_on_poisson(self):
        # 文档化行为:IC(0) 在 Poisson 上可用且通常减少迭代;不断言必然加速
        n = 30
        A = csr_from_coo(n, poisson_1d(n))
        b = [1.0] * n
        res = pcg_solve(A, b, atol=1e-10, rtol=1e-10, max_iter=500, preconditioner="ic0")
        self.assertEqual(res.status, STATUS_CONVERGED)

    def test_parameter_validation(self):
        A = csr_from_coo(2, [(0, 0, 1.0), (1, 1, 1.0)])
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0])                       # b 长度错
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, float("nan")])         # b 非有限
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], x0=[0.0])        # x0 长度错
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], atol=0.0, rtol=0.0)   # 容差全零
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], atol=-1.0)       # 负容差
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], rtol=float("inf"))
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], max_iter=-1)
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], max_iter=501)
        with self.assertRaises(ValueError):
            pcg_solve(A, [1.0, 1.0], preconditioner="lu")

    def test_breakdown_on_nonpositive_diagonal(self):
        # 对角含非正项:Jacobi 预条件器直接 breakdown
        A = csr_from_coo(2, [(0, 0, 1.0), (1, 1, -1.0)])
        res = pcg_solve(A, [1.0, 1.0], preconditioner="jacobi")
        self.assertEqual(res.status, STATUS_BREAKDOWN)


if __name__ == "__main__":
    unittest.main()
