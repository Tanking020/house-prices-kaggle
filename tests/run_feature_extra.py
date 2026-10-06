"""阶段 F · 回头补特征（批次 2）：房龄系 + 卫浴总数 —— 单个特征的 A/B 筛选。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_feature_extra.py        # 筛选：5 折单次（快，约 2 分钟）
    .\\.venv\\Scripts\\python.exe tests\\run_feature_extra.py 10     # 确认：5×10（慢，约 35 分钟）

⭐ 两段式策略（值得记住的实验习惯）：
    先用**便宜的尺子**把"明显没用"的候选筛掉，只把有希望的送去**贵的尺子**确认。
    否则就是在"等 35 分钟才发现方向不对"。

参考配方固定为当前最好：`QualArea` + `TotalSF`，模型 = `models.make_best_model()`。
每个候选特征**单独**加在参考配方上 → 逐行可归因。

⚠️ 判读标准（沿用阶段 C 的教训）：
    ① 先看"体检"里的**与目标相关性**——|r| 很小的基本可以直接扔；
    ② 再看 CV 分数变化——**小于 0.001（噪声线）视为无效**；
    ③ 单次 5 折的均值本身有 ±0.0006 的抖动，所以筛选阶段只信"变化 > 0.002"的信号。
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
from house_prices.preprocess import FEATURE_BUILDERS, make_pipeline  # noqa: E402

# 命令行第 1 个参数 = 重复次数（不写 = 1，即 5 折单次）
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

BASE = ("QualArea", "TotalSF")
CANDIDATES = ["HouseAge", "RemodAge", "IsRemodeled", "TotalBath"]


def health_check(X, y_log) -> None:
    """先做体检：缺失数 / 取值范围 / 与 log(价格) 的相关性。"""
    print("候选特征体检")
    print(f"  {'特征':<12s} {'缺失':>4s} | {'范围':>14s} | 与 log(价格) 相关性")
    print("  " + "-" * 58)
    for name in CANDIDATES:
        col = FEATURE_BUILDERS[name](X)
        n_na = int(col.isna().sum())
        filled = col.fillna(col.median())        # 只为算相关性，临时补中位数
        r = float(np.corrcoef(filled, y_log)[0, 1])
        flag = "  ← 太弱" if abs(r) < 0.10 else ""
        rng = f"{filled.min():.0f} ~ {filled.max():.0f}"
        print(f"  {name:<12s} {n_na:>4d} | {rng:>14s} | {r:+.3f}{flag}")


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)
    y_log = np.log1p(y).to_numpy()

    print(f"参考配方 {BASE}；模型 = CatBoost 最优配置；CV = 5 折 × {REPEATS}\n")
    health_check(X, y_log)

    rows = [("参考（不加新特征）", BASE)]
    rows += [(f"+ {name}", BASE + (name,)) for name in CANDIDATES]

    print(f"\nCV 结果（越小越好；噪声线 ≈ 0.001）")
    print(f"  {'方案':<22s} {'CV RMSE(log)':>14s} {'相比参考':>10s} {'用时':>8s}")
    print("  " + "-" * 60)

    ref = None
    for label, feats in rows:
        pipe = make_pipeline(make_best_model(), X, derived_features=feats)
        t0 = time.perf_counter()
        mean, std = cv_rmse_log(pipe, X, y, n_repeats=REPEATS)
        used = time.perf_counter() - t0
        if ref is None:
            ref = mean
            delta = "—"
        else:
            delta = f"{mean - ref:+.5f}"
        print(f"  {label:<22s} {mean:>8.5f}±{std:.5f} {delta:>10s} {used:>7.1f}s")


if __name__ == "__main__":
    main()
