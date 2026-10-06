"""阶段 E · E-2：联合扫描 CatBoost 的 `learning_rate × iterations`（参数耦合）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_tuning_lr_iters.py

为什么"一次只调一个参数"在这里会骗你？
- `learning_rate`（每棵树学多少）和 `iterations`（一共学多少轮）是**耦合**的：
  学习率减半 ≈ 每步只迈半步 → 通常需要**更多步**才能走到同样的位置。
- E-1 得出"depth=6 最好"，但那是在 iterations=300 固定的前提下得到的结论 → 未必是全局最优。

本脚本做两件事：
  ① 表 A：3×3 网格  lr ∈ {0.025, 0.05, 0.1} × iterations ∈ {300, 600, 1200}（depth 固定 6）
     其中带 * 的格子满足 lr × iterations ≈ 30（"学习预算"相同）→ 看它们分数是否接近
  ② 表 B：回答 E-1 留下的疑问 —— 浅树（depth=2）为什么差？
     是"复杂度不够（欠拟合）"还是"没训够（迭代次数不足）"？
     做法：把 depth=2 的 iterations 从 300 提到 1200（学习预算 ×4），看能否追回来
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
LRS = [0.025, 0.05, 0.1]
ITERS = [300, 600, 1200]
BUDGET = 30.0          # lr × iterations 的"相等学习预算"参考值


def make_model(lr: float, iters: int, depth: int = 6) -> CatBoostRegressor:
    """只把要扫的参数暴露出来，其余全部固定（对照实验的关键）。"""
    return CatBoostRegressor(
        iterations=iters,
        learning_rate=lr,
        depth=depth,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def score(X, y, lr: float, iters: int, depth: int = 6) -> tuple[float, float]:
    """跑一次 5×10 折 CV，返回 (均值, 标准差)。"""
    return cv_rmse_log(
        make_pipeline(make_model(lr, iters, depth), X, derived_features=BASE),
        X,
        y,
        n_repeats=10,
    )


def train_fit(X, y, lr: float, iters: int, depth: int = 6) -> float:
    """在【全量训练集】上 fit 后的拟合误差（诊断用，不是评估指标）。"""
    pipe = make_pipeline(make_model(lr, iters, depth), X, derived_features=BASE)
    pipe.fit(X, np.log1p(y))
    return rmse(np.log1p(y), pipe.predict(X))


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    # ---------------- 表 A：lr × iterations 网格 ----------------
    print("表 A：depth=6 固定；格子 = CV RMSE(log) ± 标准差（5×10 折）")
    print(f"      * = lr×iterations ≈ {BUDGET:g}（学习预算相同的格子，应互相可比）\n")

    header = "  lr \\ iters |" + "".join(f"{it:^19}|" for it in ITERS)
    print(header)
    print("-" * len(header))

    grid: dict[tuple[float, int], tuple[float, float]] = {}   # 元组当键
    for lr in LRS:
        cells = []
        for it in ITERS:
            mean, std = score(X, y, lr, it)
            grid[(lr, it)] = (mean, std)
            # 浮点数别用 == 比，用"差得够小"来判断（1e-6 容差）
            mark = "*" if abs(lr * it - BUDGET) < 1e-6 else " "
            cells.append(f"{mark}{mean:.5f} ± {std:.5f}".center(19))
        print(f"{lr:>11.3f} |" + "|".join(cells) + "|")

    best = min(grid, key=lambda k: grid[k][0])
    print(f"\n  → 表 A 最优：lr={best[0]:g}, iterations={best[1]}"
          f"  →  CV {grid[best][0]:.5f} ± {grid[best][1]:.5f}")

    # ---------------- 表 B：浅树是不是"没训够"？ ----------------
    print("\n表 B：为什么 depth=2 更差？（把 learning_rate 固定 0.05，只变 depth 与 iterations）\n")
    probe = [
        (2, 300),      # E-1 里的原配置（应复现 0.13242）
        (2, 1200),     # 学习预算 ×4 → 若大幅改善 = 原来只是"没训够"
        (6, 300),      # 对照组（应复现 0.12263）
        (6, 1200),     # 深树也放大预算 → 看会不会过拟合
    ]
    print("depth | iterations | CV RMSE(log)      | 训练 RMSE(log)")
    print("------+------------+-------------------+---------------")
    for depth, it in probe:
        mean, std = score(X, y, 0.05, it, depth)
        tr = train_fit(X, y, 0.05, it, depth)
        print(f"{depth:^5} | {it:^10} | {mean:.5f} ± {std:.5f} | {tr:.5f}")


if __name__ == "__main__":
    main()
