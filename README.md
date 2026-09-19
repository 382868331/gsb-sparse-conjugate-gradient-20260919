# 稀疏线性系统预条件共轭梯度求解库

纯 Python 3.14 标准库实现,Windows 原生离线可运行,无第三方依赖。
面向稀疏 SPD(对称正定)系统 A x = b 的预条件共轭梯度(PCG)求解,
以真实残差判定收敛,明确区分 converged / max_iter / breakdown 三种结局。

## 接口

### `csr_from_coo(n, triplets) -> CSRMatrix`

把 COO 三元组 `(i, j, v)` 规范化为 CSR:

- 约束 `1 <= n <= 256`,三元组数量 `<= 10000`;
- 重复 `(i, j)` 用 `math.fsum` 精确合并,合并后精确为 0 的项被去除;
- 按 (行, 列) 排序;拒绝非法索引、非数值、非有限值(`ValueError`);
- 要求**精确对称**:每个非对角项 `A[i,j]` 必须有值精确相等的 `A[j,i]`
  (缺项视为 0)。不做隐式对称化,也不稠密化。

`CSRMatrix` 提供 `matvec(x)` 与 `nnz`,不提供稠密化接口。

### `pcg_solve(A, b, x0=None, *, atol=1e-10, rtol=1e-8, max_iter=100, preconditioner="jacobi") -> PCGResult`

- `x0`:可选初值(默认零向量);
- `atol` / `rtol`:有限非负,且至少一个为正。
  收敛判据为真实残差 `||b - A x||2 <= atol + rtol * ||b||2`;
- `max_iter`:`0 <= max_iter <= 500`;
- `preconditioner`:`"jacobi"`(对角)或 `"ic0"`(零填充不完全 Cholesky)。

行为约定:

- 先算初始真实残差,初值已达标则 0 迭代收敛,不构建预条件器;
- IC(0) 只保留 A 原下三角模式,不加填充、不做主元扰动、不隐式切换;
- 每 10 次迭代及候选收敛/退出时重算真实残差;候选未达标时按真实残差
  重启搜索方向;预算耗尽返回 `max_iter`;
- 非正主元 / 非正曲率 / 非有限中间值返回 `breakdown`;
- 正定性是调用方前提,库不做完整 SPD 预检,也不保证 IC(0) 不失败或必然加速。

### `PCGResult`

| 字段 | 含义 |
| --- | --- |
| `status` | `"converged"` / `"max_iter"` / `"breakdown"` |
| `x` | 最终迭代向量 |
| `iterations` | 实际 PCG 迭代数 |
| `residual_history` | `[(迭代编号, 真实残差 ||b-Ax||2), ...]`,只记录真实残差 |
| `matvec_count` | 实际矩阵-向量乘次数(含残差复核) |
| `preconditioner` / `message` | 所用预条件器与状态说明 |

## 运行

```bat
python demo.py                                  :: 演示:正常求解 + 实际触发的失败(约 1 秒内)
python -m unittest discover -s tests -v         :: 单元测试
```

演示内容:n=64 一维 Poisson SPD 系统(已知解)分别用 Jacobi 与 IC(0)
求解并核对真实残差;对称但非正定矩阵上 IC(0) 主元为负触发 `breakdown`;
以及 `max_iter` 预算耗尽时的状态与残差记录。

## 验证方式

测试(`tests/test_sparse_pcg.py`)覆盖:重复 COO 的 fsum 合并与精确零去除、
非对称拒绝(含差一个 ulp)、缩放对角一步收敛、IC(0) 主元失败、零右端非零
初值、0 迭代初值解、max_iter(0 预算与耗尽)、真实残差记录的独立重算核对,
以及固定种子下 n<=8 随机 SPD 与独立高斯消元参考解的对照、已知解稀疏案例。
