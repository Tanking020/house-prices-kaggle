"""尺子（evaluate.py）的冒烟测试：确认它可用，并给出几个参考分数。

运行方式（在项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_evaluate.py

说明：
- 这只是"试尺子"，不是真正的建模。
- 用到的模型都是 sklearn 现成的，不需要自己实现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 让脚本能 import 到 src/house_prices（不依赖运行目录）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.dummy import DummyRegressor          # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from house_prices.data import get_xy, load_raw     # noqa: E402
from house_prices.evaluate import cv_rmse_log      # noqa: E402


def main() -> None:
    train, test = load_raw()
    print(f"train: {train.shape} | test: {test.shape}")

    X, y = get_xy(train)
    print(f"X: {X.shape} | y: {y.shape}")

    cases = [
        ("① 只猜均值（地板基线）", DummyRegressor(strategy="mean"), ["OverallQual"]),
        ("② 线性回归（OverallQual）", LinearRegression(), ["OverallQual"]),
        ("③ 线性回归（OverallQual+GrLivArea）", LinearRegression(), ["OverallQual", "GrLivArea"]),
    ]

    print("\n--- 同一把尺子给不同模型打分 ---")
    for name, model, cols in cases:
        mean, std = cv_rmse_log(model, X[cols], y)
        print(f"{name:36s} RMSE(log) = {mean:.5f} ± {std:.5f}")

    # ---- 演示：单次 5 折的"均值"本身会晃，重复 K 折把它稳住 ----
    print("\n--- 单次 5 折 vs 重复 5 折（看'切分运气'）---")
    X2 = X[["OverallQual", "GrLivArea"]]

    # 换 10 个种子各跑一次"单次 5 折"，看均值稳不稳
    single_means = [
        cv_rmse_log(LinearRegression(), X2, y, seed=s)[0] for s in range(10)
    ]
    print("单次5折·10个种子的均值:")
    print("  " + ", ".join(f"{v:.5f}" for v in single_means))
    print(
        f"  → 这 10 个均值: 平均={np.mean(single_means):.5f}"
        f"  波动(std)={np.std(single_means):.5f}"
    )

    # 重复 5 折（5x10 = 50 折）
    mean, std = cv_rmse_log(LinearRegression(), X2, y, n_repeats=10)
    print(f"重复5折(5x10, 共50折)      : {mean:.5f} ± {std:.5f}")


if __name__ == "__main__":
    main()
