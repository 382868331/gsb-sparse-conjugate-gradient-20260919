"""稀疏 SPD 线性系统的预条件共轭梯度(PCG)求解库。

仅使用 Python 标准库,Windows 原生离线可运行。

功能:
- COO 三元组 -> CSR 规范化(重复项用 math.fsum 合并、去除精确 0、排序);
- 严格逐元素对称性校验(缺项视为 0,不做隐式对称化或稠密化);
- Jacobi / IC(0) 预条件的 PCG,带周期性真实残差复核与方向重启;
- 明确的状态报告:converged / max_iter / breakdown。

正定性是调用方前提,本库不做完整 SPD 预检;数值失败通过 breakdown 状态上报。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "CSRMatrix",
    "PCGResult",
    "csr_from_coo",
    "pcg_solve",
    "STATUS_CONVERGED",
    "STATUS_MAX_ITER",
    "STATUS_BREAKDOWN",
]

N_MAX = 256
TRIPLETS_MAX = 10000
MAX_ITER_LIMIT = 500
RESIDUAL_REFRESH_INTERVAL = 10

STATUS_CONVERGED = "converged"
STATUS_MAX_ITER = "max_iter"
STATUS_BREAKDOWN = "breakdown"

_PRECONDITIONERS = ("jacobi", "ic0")


class CSRMatrix:
    """压缩稀疏行(CSR)矩阵,仅支持 matvec,不提供稠密化接口。"""

    __slots__ = ("n", "indptr", "indices", "data")

    def __init__(self, n: int, indptr: list[int], indices: list[int], data: list[float]):
        self.n = n
        self.indptr = indptr
        self.indices = indices
        self.data = data

    @property
    def nnz(self) -> int:
        return len(self.data)

    def matvec(self, x: list[float]) -> list[float]:
        n = self.n
        if len(x) != n:
            raise ValueError(f"matvec: expected vector of length {n}, got {len(x)}")
        indptr, indices, data = self.indptr, self.indices, self.data
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += data[k] * x[indices[k]]
            y[i] = s
        return y


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def csr_from_coo(n: int, triplets) -> CSRMatrix:
    """把 COO 三元组 (i, j, v) 规范化为 CSR。

    - 重复 (i, j) 用 math.fsum 精确合并,合并后精确为 0 的项被去除;
    - 按 (行, 列) 排序;
    - 拒绝非法索引、非有限值;
    - 要求精确对称:对每个非对角项 A[i,j],必须存在 A[j,i] 且值精确相等
      (缺项视为 0)。不做隐式对称化。
    """
    if not _is_int(n):
        raise ValueError(f"n must be an integer, got {n!r}")
    if not 1 <= n <= N_MAX:
        raise ValueError(f"n must satisfy 1 <= n <= {N_MAX}, got {n}")

    triplets = list(triplets)
    if len(triplets) > TRIPLETS_MAX:
        raise ValueError(f"too many triplets: {len(triplets)} > {TRIPLETS_MAX}")

    acc: dict[tuple[int, int], list[float]] = {}
    for pos, t in enumerate(triplets):
        try:
            i, j, v = t
        except (TypeError, ValueError):
            raise ValueError(f"triplet #{pos} is not a (i, j, v) triple: {t!r}") from None
        if not _is_int(i) or not _is_int(j):
            raise ValueError(f"triplet #{pos}: indices must be integers, got ({i!r}, {j!r})")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"triplet #{pos}: index ({i}, {j}) out of range for n={n}")
        if not _is_real(v):
            raise ValueError(f"triplet #{pos}: value must be a real number, got {v!r}")
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"triplet #{pos}: non-finite value {v!r}")
        acc.setdefault((i, j), []).append(v)

    merged: dict[tuple[int, int], float] = {}
    for key, vals in acc.items():
        s = math.fsum(vals)
        if s != 0.0:  # 去除精确 0
            merged[key] = s

    # 精确对称校验:缺项视为 0
    for (i, j), v in merged.items():
        if i == j:
            continue
        if merged.get((j, i), 0.0) != v:
            raise ValueError(
                f"matrix is not exactly symmetric: A[{i},{j}]={v!r} but "
                f"A[{j},{i}]={merged.get((j, i), 0.0)!r}"
            )

    entries = sorted(merged.items())
    indptr = [0] * (n + 1)
    indices: list[int] = []
    data: list[float] = []
    row = 0
    for (i, j), v in entries:
        while row < i:
            row += 1
            indptr[row] = len(data)
        indices.append(j)
        data.append(v)
    for r in range(row + 1, n + 1):
        indptr[r] = len(data)
    return CSRMatrix(n, indptr, indices, data)


def _norm2(v: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in v))


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


class _Breakdown(Exception):
    """内部信号:数值失败,转换为 breakdown 状态返回。"""


class _Jacobi:
    """Jacobi(对角)预条件。对角缺失或非正 => breakdown。"""

    def __init__(self, A: CSRMatrix):
        n = A.n
        diag = [0.0] * n
        for i in range(n):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                if A.indices[k] == i:
                    diag[i] = A.data[k]
                    break
        for i, d in enumerate(diag):
            if not math.isfinite(d) or d <= 0.0:
                raise _Breakdown(f"jacobi: non-positive or missing diagonal at row {i} (A[i,i]={d!r})")
        self._inv = [1.0 / d for d in diag]

    def solve(self, r: list[float]) -> list[float]:
        return [ri * di for ri, di in zip(r, self._inv)]


class _IC0:
    """IC(0) 不完全 Cholesky 预条件。

    只保留 A 原下三角的非零模式,不加填充、不做主元扰动、不隐式切换。
    遇到非正或非有限主元 => breakdown。
    """

    def __init__(self, A: CSRMatrix):
        n = A.n
        # 逐行构建 L 的严格下三角部分(行 -> {列: 值})及对角
        l_rows: list[dict[int, float]] = []
        diag = [0.0] * n
        for i in range(n):
            a_lower = {}
            for k in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[k]
                if j <= i:
                    a_lower[j] = A.data[k]
            row: dict[int, float] = {}
            for j in sorted(a_lower):
                s = a_lower[j]
                lj = row if j == i else l_rows[j]
                # 仅对双方模式中共同存在的 k<j 做消去(保持 IC(0) 模式)
                for kcol, lik in row.items():
                    if kcol >= j:
                        break
                    ljk = lj.get(kcol)
                    if ljk is not None:
                        s -= lik * ljk
                if i == j:
                    if not math.isfinite(s) or s <= 0.0:
                        raise _Breakdown(
                            f"ic0: non-positive pivot at row {i} (pivot={s!r})"
                        )
                    diag[i] = math.sqrt(s)
                else:
                    row[j] = s / diag[j]
            l_rows.append(row)
        # 前代需要 L 的 CSR(行),回代需要 L^T 的 CSR(即 L 的列)
        self._n = n
        self._diag = diag
        self._l_rows = l_rows
        lt_rows: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for j, v in l_rows[i].items():
                lt_rows[j].append((i, v))
        self._lt_rows = lt_rows

    def solve(self, r: list[float]) -> list[float]:
        n = self._n
        diag = self._diag
        # 前代: L y = r
        y = [0.0] * n
        for i in range(n):
            s = r[i]
            for j, lij in self._l_rows[i].items():
                s -= lij * y[j]
            y[i] = s / diag[i]
        # 回代: L^T z = y
        z = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i]
            for j, lji in self._lt_rows[i]:
                s -= lji * z[j]
            z[i] = s / diag[i]
        return z


@dataclass
class PCGResult:
    """PCG 求解结果。

    status: "converged" | "max_iter" | "breakdown"
    iterations: 实际执行的 PCG 迭代数
    residual_history: [(迭代编号, 该处真实残差 ||b-Ax||2), ...],只记录真实残差
    matvec_count: 实际执行的矩阵-向量乘次数(含残差复核)
    """

    status: str
    x: list[float]
    iterations: int
    residual_history: list[tuple[int, float]]
    matvec_count: int
    preconditioner: str
    message: str = ""

    @property
    def converged(self) -> bool:
        return self.status == STATUS_CONVERGED


def _validate_vector(name: str, v, n: int) -> list[float]:
    try:
        vals = list(v)
    except TypeError:
        raise ValueError(f"{name} must be a sequence of {n} finite reals") from None
    if len(vals) != n:
        raise ValueError(f"{name} must have length {n}, got {len(vals)}")
    out = []
    for idx, x in enumerate(vals):
        if not _is_real(x) or not math.isfinite(float(x)):
            raise ValueError(f"{name}[{idx}] is not a finite real: {x!r}")
        out.append(float(x))
    return out


def _validate_tolerance(name: str, v) -> float:
    if not _is_real(v):
        raise ValueError(f"{name} must be a real number, got {v!r}")
    v = float(v)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {v!r}")
    return v


def pcg_solve(
    A: CSRMatrix,
    b,
    x0=None,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    max_iter: int = 100,
    preconditioner: str = "jacobi",
) -> PCGResult:
    """用预条件共轭梯度法解 A x = b(A 为调用方保证的 SPD 矩阵)。

    收敛判据: ||b - A x||2 <= atol + rtol * ||b||2(真实残差,非递推残差)。
    每 10 次迭代及候选收敛/退出时重算真实残差;候选未达标时按真实残差
    重启搜索方向。预算耗尽返回 "max_iter";非正主元/非正曲率/非有限中间值
    返回 "breakdown"。
    """
    if not isinstance(A, CSRMatrix):
        raise ValueError("A must be a CSRMatrix built by csr_from_coo")
    n = A.n
    b = _validate_vector("b", b, n)
    x = [0.0] * n if x0 is None else _validate_vector("x0", x0, n)

    atol = _validate_tolerance("atol", atol)
    rtol = _validate_tolerance("rtol", rtol)
    if atol == 0.0 and rtol == 0.0:
        raise ValueError("at least one of atol/rtol must be positive")
    if not _is_int(max_iter):
        raise ValueError(f"max_iter must be an integer, got {max_iter!r}")
    if not 0 <= max_iter <= MAX_ITER_LIMIT:
        raise ValueError(f"max_iter must satisfy 0 <= max_iter <= {MAX_ITER_LIMIT}, got {max_iter}")
    if preconditioner not in _PRECONDITIONERS:
        raise ValueError(f"preconditioner must be one of {_PRECONDITIONERS}, got {preconditioner!r}")

    matvec_count = 0

    def true_residual(xv: list[float]) -> tuple[list[float], float]:
        nonlocal matvec_count
        ax = A.matvec(xv)
        matvec_count += 1
        r = [bi - ai for bi, ai in zip(b, ax)]
        return r, _norm2(r)

    def result(status: str, iterations: int, history, message: str) -> PCGResult:
        return PCGResult(
            status=status,
            x=list(x),
            iterations=iterations,
            residual_history=history,
            matvec_count=matvec_count,
            preconditioner=preconditioner,
            message=message,
        )

    tol = atol + rtol * _norm2(b)

    # 先算真实残差:初值已满足则 0 迭代收敛,不构建预条件器
    r, norm_r = true_residual(x)
    history: list[tuple[int, float]] = [(0, norm_r)]
    if norm_r <= tol:
        return result(STATUS_CONVERGED, 0, history, "initial guess satisfies tolerance")

    # 初值未达标,此时才构建预条件器
    try:
        M = _Jacobi(A) if preconditioner == "jacobi" else _IC0(A)
    except _Breakdown as e:
        return result(STATUS_BREAKDOWN, 0, history, str(e))

    def fresh_direction(rv: list[float]) -> tuple[list[float], float]:
        """z = M^{-1} r,返回 (z, r·z);非有限或非正 => breakdown。"""
        z = M.solve(rv)
        rz = _dot(rv, z)
        if not math.isfinite(rz) or rz <= 0.0:
            raise _Breakdown(f"non-positive or non-finite preconditioned residual (r·z={rz!r})")
        return z, rz

    k = 0
    try:
        z, rz = fresh_direction(r)
        p = list(z)
        while k < max_iter:
            ap = A.matvec(p)
            matvec_count += 1
            pap = _dot(p, ap)
            if not math.isfinite(pap) or pap <= 0.0:
                raise _Breakdown(f"non-positive or non-finite curvature p^T A p = {pap!r} at iteration {k + 1}")
            alpha = rz / pap
            if not math.isfinite(alpha):
                raise _Breakdown(f"non-finite step length at iteration {k + 1}")
            for i in range(n):
                x[i] += alpha * p[i]
                r[i] -= alpha * ap[i]
            k += 1
            rec_norm = _norm2(r)
            candidate = rec_norm <= tol
            # 每 10 次迭代、候选收敛或预算耗尽时,重算真实残差
            if k % RESIDUAL_REFRESH_INTERVAL == 0 or candidate or k == max_iter:
                r, norm_r = true_residual(x)
                if not math.isfinite(norm_r):
                    raise _Breakdown(f"non-finite true residual at iteration {k}")
                history.append((k, norm_r))
                if norm_r <= tol:
                    return result(STATUS_CONVERGED, k, history, "true residual within tolerance")
                if k == max_iter:
                    break
                if candidate:
                    # 候选未达标:用真实残差重启搜索方向
                    z, rz = fresh_direction(r)
                    p = list(z)
                    continue
                # 周期复核:以真实残差替换递推残差,方向照常做 beta 更新
            else:
                if not math.isfinite(rec_norm):
                    raise _Breakdown(f"non-finite recursive residual at iteration {k}")
            z, rz_new = fresh_direction(r)
            beta = rz_new / rz
            for i in range(n):
                p[i] = z[i] + beta * p[i]
            rz = rz_new
    except _Breakdown as e:
        return result(STATUS_BREAKDOWN, k, history, str(e))

    return result(
        STATUS_MAX_ITER,
        k,
        history,
        f"iteration budget exhausted (max_iter={max_iter}); last true residual {history[-1][1]!r} > tolerance {tol!r}",
    )
