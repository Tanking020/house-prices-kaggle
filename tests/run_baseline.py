"""B4 · 基线实验：用统一的尺子给"地板"和"Ridge 流水线"打分。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_baseline.py

注意：所有 `fit` 都发生在 Pipeline 内部、且只在每一折的训练部分上，
     因此不存在数据泄漏。
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw        # noqa: E402
from house_prices.evaluate import cv_rmse_log         # noqa: E402
from house_prices.preprocess import make_pipeline     # noqa: E402


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    print(f"数据: X {X.shape}, y {y.shape}")
    print("\n模型                          CV (RMSE-log, 越小越好)")
    print("-" * 58)

    # ① 地板：只猜均值（不看任何特征）
    m0, s0 = cv_rmse_log(DummyRegressor(strategy="mean"), X, y)
    print(f"① 地板（只猜均值）            {m0:.5f} ± {s0:.5f}")

    # ② 基线：完整预处理流水线 + Ridge
    ridge = make_pipeline(Ridge(alpha=1.0), X)
    m1, s1 = cv_rmse_log(ridge, X, y)
    print(f"② Ridge(alpha=1)，5 折        {m1:.5f} ± {s1:.5f}")

    # ③ 用重复 5 折拿到更稳的参考值
    m2, s2 = cv_rmse_log(ridge, X, y, n_repeats=10)
    print(f"③ Ridge(alpha=1)，5×10        {m2:.5f} ± {s2:.5f}   ← 更稳的参考值")

    print(f"\n相比地板改善: {m0 - m2:.5f}（{(m0 - m2) / m0 * 100:.1f}%）")


if __name__ == "__main__":
    main()
