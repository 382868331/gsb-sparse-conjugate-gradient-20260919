"""Demo: PCG with Jacobi and IC(0) on a sparse SPD system, plus real
numerical failures (IC(0) pivot breakdown and non-positive curvature
on an indefinite matrix).

Run:  python demo.py
Everything printed below is computed by the library at runtime.
"""

import math
import time

from sparse_pcg import csr_from_coo, solve_pcg


def poisson_2d(m):
    """SPD 5-point Laplacian on an m x m grid (n = m*m) as COO triplets."""
    t = []

    def idx(r, c):
        return r * m + c

    for r in range(m):
        for c in range(m):
            i = idx(r, c)
            t.append((i, i, 4.0))
            for rr, cc in ((r - 1, c), (r, c - 1)):
                if 0 <= rr < m and 0 <= cc < m:
                    j = idx(rr, cc)
                    t.append((i, j, -1.0))
                    t.append((j, i, -1.0))
    return t


def main():
    t0 = time.perf_counter()

    # --- Normal case: sparse SPD system with a known solution ----------
    m = 11
    n = m * m
    A = csr_from_coo(n, poisson_2d(m))
    x_true = [math.sin(2.0 * math.pi * i / (n - 1)) for i in range(n)]
    b = A.matvec(x_true)
    print(f"[1] 2D Poisson SPD system ({m}x{m} grid), n={n}, nnz={A.nnz}, "
          f"known solution")
    print(f"    target: ||b-Ax||2 <= atol + rtol*||b||2 "
          f"with atol=rtol=1e-10")
    for pre in ("jacobi", "ic0"):
        res = solve_pcg(A, b, atol=1e-10, rtol=1e-10, max_iter=500,
                        preconditioner=pre)
        err = max(abs(g - t) for g, t in zip(res.x, x_true))
        it, rn = res.residual_history[-1]
        print(f"    {pre:6s}: status={res.status:9s} iters={res.iterations:3d} "
              f"matvecs={res.matvecs:3d} true|b-Ax|={rn:.3e} "
              f"max|x-x_true|={err:.3e}")
        hist = " ".join(f"({i},{r:.1e})" for i, r in res.residual_history[:4])
        print(f"           true-residual records: {hist} ... ({it},{rn:.1e})")

    # --- Real failure 1: IC(0) on a symmetric but indefinite matrix ----
    # A = [[1, 2], [2, 1]] is symmetric; its IC(0) pivot at row 1 is
    # 1 - 2**2 = -3 <= 0, so the factorization breaks down for real.
    bad = csr_from_coo(2, [(0, 0, 1.0), (0, 1, 2.0),
                           (1, 0, 2.0), (1, 1, 1.0)])
    res = solve_pcg(bad, [1.0, 1.0], preconditioner="ic0")
    print(f"\n[2] Symmetric but indefinite matrix [[1,2],[2,1]], IC(0)")
    print(f"    status={res.status} iters={res.iterations} "
          f"matvecs={res.matvecs}")
    print(f"    message: {res.message}")
    print(f"    true-residual records: {res.residual_history}")

    # --- Real failure 2: non-positive curvature during PCG -------------
    # Same indefinite matrix with Jacobi. b = [1, -1] is an eigenvector
    # with eigenvalue -1, so p^T A p = -2 < 0 on the first step.
    res = solve_pcg(bad, [1.0, -1.0], preconditioner="jacobi")
    print(f"\n[3] Same indefinite matrix, Jacobi, b=[1,-1]")
    print(f"    status={res.status} iters={res.iterations} "
          f"matvecs={res.matvecs}")
    print(f"    message: {res.message}")

    print(f"\nelapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
