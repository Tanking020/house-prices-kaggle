"""阶段 E · E-1：单变量扫描 CatBoost 的 `depth`（树深度 = 模型复杂度）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_tuning_depth.py

为什么要先扫 depth？
- #16/#17 已证明：树太深 → 小数据（1168 行）过拟合。depth 就是"复杂度"这根轴。
- 一次只动一个参数 → 趋势可归因。其余全部固定（iterations=300, lr=0.05, 特征不变）。

本脚本同时打印【训练集 RMSE】，用来"看见"过拟合：
    depth 变大 → 训练误差一直降（模型越来越会"背"）
    但 CV 误差是 U 形（先降后升）→ 最低点才是"复杂度刚好"的位置
⚠️ 训练集 RMSE 不是评估指标，只是诊断用（评估必须用 CV）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw        # noqa: E402
from house_prices.evaluate import cv_rmse_log, rmse   # noqa: E402
from house_prices.preprocess import make_pipeline     # noqa: E402

BASE = ("QualArea", "TotalSF")
DEPTHS = [2, 3, 4, 5, 6, 7, 8]


def make_model(depth: int) -> CatBoostRegressor:
    """把"只变 depth"这件事写成函数，避免各处参数抄漏。"""
    return CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=depth,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)
    y_log = np.log1p(y)

    print(f"特征固定 {BASE}；CatBoost iterations=300, lr=0.05；尺子 = 5×10 折\n")
    print(f"{'depth':>5} | {'CV RMSE(log)':>13} | {'CV 标准差':>9} | {'训练 RMSE(log)':>14}")
    print("-" * 54)

    for d in DEPTHS:
        # ① CV 分数（唯一可信的评估）
        mean, std = cv_rmse_log(
            make_pipeline(make_model(d), X, derived_features=BASE),
            X,
            y,
            n_repeats=10,
        )

        # ② 训练集拟合误差：看"模型有多会背"（诊断用）
        pipe = make_pipeline(make_model(d), X, derived_features=BASE)
        pipe.fit(X, y_log)
        tr = rmse(y_log, pipe.predict(X))

        print(f"{d:>5} | {mean:>13.5f} | {std:>9.5f} | {tr:>14.5f}")


if __name__ == "__main__":
    main()
