# 稀疏线性系统预条件共轭梯度求解库

纯 Python 3.14 标准库实现（无第三方依赖），在 Windows 原生环境离线运行。
功能：COO → CSR 稀疏矩阵装配，以及带 Jacobi / IC(0) 预条件的共轭梯度法（PCG）
求解对称正定（SPD）稀疏线性方程组 `A x = b`。收敛判定始终使用**真实残差**
`||b - A x||2`，不把递推残差冒充真实残差；数值失败以状态返回而非抛异常。

## 环境与命令

- Windows 原生 Python 3.14.7，仅标准库，无安装步骤、无外部服务。
- 演示：`python demo.py`（约 1 秒内完成，展示正常结果与实际触发的失败）
- 测试：`python -m unittest discover -s tests -v`

## 接口

### `csr_from_coo(n, triplets) -> CSRMatrix`

从 COO 三元组 `(i, j, v)` 装配对称 CSR 矩阵：

- `1 <= n <= 256`，三元组数量 `<= 10000`；
- 重复项用 `math.fsum` 精确合并，合并后的精确 0 被删除，按 (row, col) 排序规范化；
- 非法索引（非整数 / 越界）与非有限值（NaN/inf）抛出 `SparseMatrixError`；
- 合并后要求 `A[i,j] == A[j,i]` 精确相等（缺项视为 0.0），否则抛出
  `SparseMatrixError`；不做隐式对称化或稠密化。

`CSRMatrix` 提供 `n`、`nnz`、`matvec(x)`、`diagonal()`。

### `solve_pcg(A, b, x0=None, *, atol=1e-10, rtol=1e-8, max_iter=500, preconditioner="jacobi") -> PCGResult`

- `A`：`csr_from_coo` 返回的 CSRMatrix。**正定性是调用者前提**，不做完整预检。
- `b` / `x0`：长度 `n` 的有限实数序列；`x0` 缺省为零向量。
- `atol` / `rtol`：有限非负，且至少一个为正。收敛判据为真实残差
  `||b - A x||2 <= atol + rtol * ||b||2`。
- `max_iter`：整数，`0 <= max_iter <= 500`。
- `preconditioner`：`"jacobi"`（对角）或 `"ic0"`（IC(0) 不完全 Cholesky：
  只保留原下三角模式，不加填充、不扰动、不隐式切换；不保证不失败或一定加速）。

求解流程：先计算真实残差，若初值已满足容差则返回 0 迭代 `converged`，
否则才构建预条件。每 10 次迭代及候选收敛/退出时重算真实残差；候选未达标时
按真实残差重启搜索方向；预算耗尽返回 `max_iter`。

### `PCGResult`

- `x`：解向量（list of float）；
- `status`：`"converged"` / `"max_iter"` / `"breakdown"`（另有 `converged` 属性）；
- `iterations`：实际迭代次数；
- `residual_history`：`[(迭代编号, 真实残差范数), ...]`，全部为重算的真实残差；
- `matvecs`：实际矩阵-向量乘积次数（含真实残差重算）；
- `preconditioner`、`message`（失败原因说明）。

`breakdown` 触发条件：IC(0) 非正/非有限主元、Jacobi 非正对角、
非正曲率 `p^T A p <= 0`、非有限中间值。

## 验证方式

- `tests/test_sparse_pcg.py` 内含独立的稠密高斯消元参考（部分主元，n <= 8），
  用固定种子（`random.Random(20260920)`）生成少量小型随机对角占优 SPD 矩阵，
  两种预条件的 PCG 解均与参考解对比（误差 < 1e-6）；
- 已知解稀疏案例（Poisson 矩阵，`x_true` 已知，`b = A x_true`）；
- 失败案例：IC(0) 主元失败、Jacobi 非正对角、非正曲率；
- 覆盖：重复 COO 合并（含 fsum 精度与精确 0 删除）、非对称拒绝、缩放对角、
  零右端非零初值、0 迭代初值解、`max_iter` 耗尽、真实残差记录核对、参数校验。

## 文件

- `sparse_pcg.py` — 库本体（唯一模块）；
- `demo.py` — 演示：2D Poisson SPD 系统两种预条件求解 + 两个真实触发的失败；
- `tests/test_sparse_pcg.py` — unittest 测试集。
