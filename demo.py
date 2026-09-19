"""sparse_pcg 演示:一个正常求解(两种预条件)和一个实际触发的数值失败。

运行: python demo.py   (纯标准库,数秒内完成)
"""

import math

from sparse_pcg import csr_from_coo, pcg_solve


def poisson_1d(n):
    t = []
    for i in range(n):
        t.append((i, i, 2.0))
        if i > 0:
            t.append((i, i - 1, -1.0))
        if i < n - 1:
            t.append((i, i + 1, -1.0))
    return t


def matvec_dense_from_csr(A, x):
    return A.matvec(x)


def true_residual_norm(A, b, x):
    r = [bi - ai for bi, ai in zip(b, A.matvec(x))]
    return math.sqrt(math.fsum(v * v for v in r))


def main():
    print("=" * 72)
    print("案例 1: 一维 Poisson SPD 系统 (n=64), 已知解, Jacobi 与 IC(0) 预条件")
    print("=" * 72)
    n = 64
    A = csr_from_coo(n, poisson_1d(n))
    x_true = [math.sin(0.3 * i) + 1.0 for i in range(n)]
    b = A.matvec(x_true)
    print(f"矩阵: {n}x{n} 三对角 SPD, nnz={A.nnz}; 右端 b = A * x_true")
    for pre in ("jacobi", "ic0"):
        res = pcg_solve(A, b, atol=1e-10, rtol=1e-10, max_iter=500, preconditioner=pre)
        err = max(abs(xi - ti) for xi, ti in zip(res.x, x_true))
        rnorm = true_residual_norm(A, b, res.x)
        print(f"\n预条件器: {pre}")
        print(f"  状态            : {res.status} ({res.message})")
        print(f"  迭代次数        : {res.iterations}")
        print(f"  实际 matvec 次数: {res.matvec_count}")
        print(f"  真实残差 ||b-Ax||: {rnorm:.3e}  (独立重算, 与记录末条一致: "
              f"{abs(rnorm - res.residual_history[-1][1]) < 1e-12})")
        print(f"  与已知解最大误差: {err:.3e}")
        print(f"  真实残差记录(迭代号, 残差): "
              f"{[(k, f'{v:.2e}') for k, v in res.residual_history[:4]]} ...")

    print()
    print("=" * 72)
    print("案例 2: 实际触发的失败 —— 对称但非正定矩阵上的 IC(0) 主元失败")
    print("=" * 72)
    # A = [[1, 2], [2, 1]]: 对称,但 IC(0) 第二个主元为 1 - 4 = -3 <= 0。
    # 正定性是调用方前提,本库不预检,而是以 breakdown 状态如实上报。
    A_bad = csr_from_coo(2, [(0, 0, 1.0), (1, 1, 1.0), (0, 1, 2.0), (1, 0, 2.0)])
    res = pcg_solve(A_bad, [1.0, 1.0], preconditioner="ic0")
    print(f"矩阵 [[1, 2], [2, 1]] (对称, 非 SPD), 预条件器 ic0")
    print(f"  状态  : {res.status}")
    print(f"  信息  : {res.message}")
    print(f"  迭代数: {res.iterations}, matvec 次数: {res.matvec_count}")

    print()
    print("=" * 72)
    print("案例 3: 迭代预算耗尽 —— max_iter 状态与真实残差记录")
    print("=" * 72)
    res = pcg_solve(A, b, atol=1e-12, rtol=1e-12, max_iter=3, preconditioner="jacobi")
    print(f"同一 Poisson 系统, max_iter=3")
    print(f"  状态  : {res.status} ({res.message})")
    print(f"  真实残差记录: {[(k, f'{v:.3e}') for k, v in res.residual_history]}")


if __name__ == "__main__":
    main()
