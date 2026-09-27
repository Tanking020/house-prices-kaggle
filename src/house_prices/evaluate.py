"""统一的评估尺子：K 折交叉验证 + log 空间 RMSE。

★ 这是全项目最重要的一个文件 ★ —— 之后每一次实验都用它打分，
  保证"改了什么 → 分数怎么变"是可信的（同一把尺子）。

固定约定（不要随意改）：
- 目标先做 log1p 变换，模型在 log 空间训练
- 指标 = RMSE( log1p(y_true), pred_log )
- 默认 5 折交叉验证，随机种子 42
- 重要决策时用 `n_repeats=10`（重复 5 折 = 共 50 折），让"均值"更稳
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, RepeatedKFold

SEED = 42


def rmse(y_true, y_pred) -> float:
    """普通 RMSE（均方根误差）。越小越好。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cv_rmse_log(
    model,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = SEED,
    verbose: bool = False,
) -> tuple[float, float]:
    """K 折（或重复 K 折）交叉验证，返回 log 空间 RMSE 的 (均值, 标准差)。

    参数
    ----
    model     : 任何 sklearn 兼容模型（有 fit / predict）
    X         : 特征，pandas DataFrame 或 ndarray
    y         : 目标，**原始价格**（函数内部自动做 log1p）
    n_repeats : 1  → 普通 5 折（快，日常快速对比）
                >1 → 重复 K 折（换 n_repeats 套切分，共 n_splits*n_repeats 折；
                     更稳，用于重要决策）

    用法
    ----
    >>> mean, std = cv_rmse_log(LinearRegression(), X, y)            # 普通 5 折
    >>> mean, std = cv_rmse_log(LinearRegression(), X, y, n_repeats=10)  # 5x10

    ⚠️ 注意：重复 K 折的 `std` 是"所有折之间"的波动，和普通 5 折的 `std`
       不是同一个含义（重复时同一批样本会被多次验证）。**比均值，别直接比 std。**
    """
    y = np.asarray(y, dtype=float)

    if n_repeats == 1:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )

    scores: list[float] = []
    for i, (tr_idx, va_idx) in enumerate(splitter.split(X), start=1):
        m = clone(model)                      # 每折用全新模型，避免状态泄漏
        m.fit(_take_rows(X, tr_idx), np.log1p(y[tr_idx]))   # 在 log 空间训练
        pred_log = m.predict(_take_rows(X, va_idx))         # 模型输出在 log 空间
        s = rmse(np.log1p(y[va_idx]), pred_log)             # 在 log 空间比较
        scores.append(s)
        if verbose:
            print(f"  split {i:3d}: RMSE(log) = {s:.5f}")

    return float(np.mean(scores)), float(np.std(scores))


def _take_rows(X, idx):
    """同时支持 pandas DataFrame 与 ndarray 的按行取子集。"""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]
