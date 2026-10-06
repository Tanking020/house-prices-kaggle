"""阶段 F · 补特征：离群点验证 —— 剔除异常样本到底值不值（用 CV 说话）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_outlier_check.py      # 5 折单次（约 1 分钟）
    .\\.venv\\Scripts\\python.exe tests\\run_outlier_check.py 10   # 5×10 确认

背景（EDA 结论）：`Id` 524 / 1299 —— `GrLivArea` 分别 4676 / 5642（远大于其他房子），
但售价只有 184,750 / 160,000：**又大又便宜**，明显是异常样本。

⚠️ 为什么不能"看到就删"？
- 删掉样本 = 改变训练分布：**可能让模型在"正常房子"上更准，也可能整体变差**。
- 到底值不值，**只有 CV 说了算** —— 本脚本就是干这件事的。
- 📌 诚实声明：剔除样本后 CV 的样本集变了，所以与"参考"**不是严格同一把尺子**；
  这是业界通用做法，但要把"差异 < 0.001 视为无影响"这条记住。

参考配置 = `models.py` 里的当前最好（CatBoost + QualArea + TotalSF + Neighborhood 目标编码）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw                     # noqa: E402
from house_prices.evaluate import cv_rmse_log, rmse                 # noqa: E402
from house_prices.models import (                                   # noqa: E402
    BEST_DERIVED_FEATURES,
    BEST_TARGET_ENCODE,
    make_best_model,
)
from house_prices.preprocess import make_pipeline                   # noqa: E402

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def cv_train_dropout(X, y, drop_mask=None, n_splits=5, seed=42):
    """**严格可比**的做法：只在【训练折】里剔除离群点，验证折保持全量。

    为什么必须这样？
    - 如果连验证集一起剔，考的“卷子”就变了 —— 而那两行正是“最难预测”的样本，
      删掉它们分数当然好看，但这**不是模型变好了**。
    - 只从训练折剔除 → 两边考卷完全一样，差异只来自“模型有没有被带偏”。
    """
    y_log = np.log1p(y).to_numpy()
    dropped = np.zeros(len(X), dtype=bool) if drop_mask is None else drop_mask.to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        tr_idx = tr_idx[~dropped[tr_idx]]        # ← 关键：只从训练折里去掉
        pipe = make_pipeline(
            make_best_model(), X,
            derived_features=BEST_DERIVED_FEATURES,
            target_encode=BEST_TARGET_ENCODE,
        )
        pipe.fit(X.iloc[tr_idx], y_log[tr_idx])
        scores.append(rmse(y_log[va_idx], pipe.predict(X.iloc[va_idx])))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    outlier_mask = train["Id"].isin([524, 1299])

    print(f"当前最好配置：CatBoost + {BEST_DERIVED_FEATURES} + 目标编码 {BEST_TARGET_ENCODE}")
    print(f"CV = 5 折（seed=42，单次；噪声线 ≈ 0.001）\n")
    print(f"  {'方案':<32s} {'训练行':>6s} {'CV RMSE(log)':>15s} {'相比参考':>10s}")
    print("  " + "-" * 72)

    # ① 参考：什么都不剔
    t0 = time.perf_counter()
    ref, ref_std = cv_train_dropout(X, y, None)
    print(f"  {'① 参考（全量训练、全量验证）':<32s} {len(X):>6d} "
          f"{ref:>9.5f}±{ref_std:.5f} {'—':>10s}   （{time.perf_counter() - t0:.1f}s）")

    # ② 严格可比：只在训练折剔除
    t0 = time.perf_counter()
    strict, strict_std = cv_train_dropout(X, y, outlier_mask)
    print(f"  {'② 只在训练折剔除（严格可比）':<32s} {len(X) - 2:>6d} "
          f"{strict:>9.5f}±{strict_std:.5f} {strict - ref:>+10.5f}   "
          f"（{time.perf_counter() - t0:.1f}s）")

    # ③ 整体剔除（连考卷一起换 —— 不可直接比）
    keep = ~outlier_mask
    pipe = make_pipeline(
        make_best_model(), X.loc[keep],
        derived_features=BEST_DERIVED_FEATURES,
        target_encode=BEST_TARGET_ENCODE,
    )
    t0 = time.perf_counter()
    naive, naive_std = cv_rmse_log(pipe, X.loc[keep], y.loc[keep], n_repeats=REPEATS)
    print(f"  {'③ 连验证集一起剔除（❌ 不可比）':<32s} {len(X) - 2:>6d} "
          f"{naive:>9.5f}±{naive_std:.5f} {naive - ref:>+10.5f}   "
          f"（{time.perf_counter() - t0:.1f}s）")

    print("\n  ⚠️ 对照 ② 与 ③：如果 ③ 的“改善”远大于 ②，说明那份“改善”主要来自")
    print("     **删掉了最难预测的考卷题目**，而不是模型真的变好了。")
    print("  📌 结论要用 ② 来说话（考卷不变，只改训练）。")


if __name__ == "__main__":
    main()
