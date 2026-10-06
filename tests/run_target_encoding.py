"""阶段 F · 补特征：目标编码（OOF + 平滑）—— 本阶段最有含金量的一个特征工程。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_target_encoding.py       # 筛选：5 折单次（快，约 1.5 分钟）
    .\\.venv\\Scripts\\python.exe tests\\run_target_encoding.py 10    # 确认：5×10（慢，约 20 分钟）

为什么要做目标编码？（它解决一个"树模型效率"问题）
- 树当然能用 One-Hot 后的 `Neighborhood` 列去切分，但 **25 个 0/1 列**要表达出
  "这个街区整体贵不贵"，需要**很多次**切分才凑得出来 → 效率极低。
- 目标编码把它压成**一个数值**（"该街区的平均 log 房价"）→ 树**一次切分**就能用上。

⚠️ 但它用了 y，所以必须 **K 折 OOF + 平滑**，否则就是最经典的泄漏（见 `OofTargetEncoder`）。
   ⚠️ 判断有没有泄漏的办法：如果"训练集分数突然变得特别好、CV 却变差" → 立刻查这里。

参考配方固定为当前最好：`QualArea` + `TotalSF`，模型 = `models.make_best_model()`。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw                    # noqa: E402
from house_prices.evaluate import cv_rmse_log                      # noqa: E402
from house_prices.models import make_best_model                    # noqa: E402
from house_prices.preprocess import make_pipeline                  # noqa: E402

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
BASE = ("QualArea", "TotalSF")

CASES = [
    ("参考（不做目标编码）", ()),
    ("+ 目标编码 Neighborhood", ("Neighborhood",)),
    ("+ 目标编码 + MSSubClass", ("Neighborhood", "MSSubClass")),
]


def report_cardinality(X, y) -> None:
    """看一眼候选列：有多少个类别？每类多少样本？（说明为什么需要平滑）"""
    print("候选列的类别分布（说明为什么必须平滑）")
    for col in ("Neighborhood", "MSSubClass"):
        counts = X[col].value_counts()
        print(f"  {col:<14s} 共 {len(counts):>2d} 类 | "
              f"最少 {int(counts.min())} 条、最多 {int(counts.max())} 条、"
              f"中位数 {int(counts.median())} 条")
    print("  ⚠️ 样本最少的类别只有个位数 → 直接取均值会极不稳定 → 必须向全局均值收缩\n")


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    print(f"参考配方 {BASE}；模型 = CatBoost 最优配置；CV = 5 折 × {REPEATS}\n")
    report_cardinality(X, y)

    print(f"CV 结果（越小越好；噪声线 ≈ 0.001）")
    print(f"  {'方案':<26s} {'CV RMSE(log)':>14s} {'相比参考':>10s} {'用时':>8s}")
    print("  " + "-" * 64)

    ref = None
    for label, te_cols in CASES:
        pipe = make_pipeline(
            make_best_model(), X,
            derived_features=BASE,
            target_encode=te_cols,
        )
        t0 = time.perf_counter()
        mean, std = cv_rmse_log(pipe, X, y, n_repeats=REPEATS)
        used = time.perf_counter() - t0
        delta = "—" if ref is None else f"{mean - ref:+.5f}"
        if ref is None:
            ref = mean
        print(f"  {label:<26s} {mean:>8.5f}±{std:.5f} {delta:>10s} {used:>7.1f}s")


if __name__ == "__main__":
    main()
