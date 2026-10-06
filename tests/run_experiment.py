"""阶段 C · 特征工程实验：A/B 对比（基线 vs 加特征）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_experiment.py

纪律：一次只加一个特征；用 5×10 打分；结果记入 docs/experiments.md。
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw                # noqa: E402
from house_prices.evaluate import cv_rmse_log                 # noqa: E402
from house_prices.preprocess import (                         # noqa: E402
    FEATURE_BUILDERS,
    build_preprocessor,
    make_pipeline,
)


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    # 先冒烟：确认派生特征真的进入了流水线
    # 注意：get_feature_names_out 需要先 fit（One-Hot 的类别清单是 fit 时才确定的）
    prep = build_preprocessor(X, derived_features=("HasPool",))
    prep.fit(X)
    feat_names = prep.get_feature_names_out()
    print(f"启用 HasPool 后列数 = {len(feat_names)}，"
          f"'HasPool' 在输出里? {'HasPool' in feat_names}")
    print()

    # 参考 = 目前最好的配置；每次**只在它基础上加一个**特征 → 逐行可归因
    BASE = ("QualArea",)
    cases = {
        "参考：QualArea": BASE,
        "+ HasPool": BASE + ("HasPool",),
        "+ Has2ndFlr": BASE + ("Has2ndFlr",),
        "+ HasBsmt": BASE + ("HasBsmt",),
        "+ HasGarage": BASE + ("HasGarage",),
        "+ HasFireplace": BASE + ("HasFireplace",),
    }

    print("模型                           CV (RMSE-log, 5×10)")
    print("-" * 58)
    for name, feats in cases.items():
        model = make_pipeline(Ridge(alpha=1.0), X, derived_features=feats)
        mean, std = cv_rmse_log(model, X, y, n_repeats=10)
        print(f"{name:26s} {mean:.5f} ± {std:.5f}")

    # ---- 诊断：这些 0/1 标志的分布（越接近 0 或 1 越没用）----
    print("\n标志的 1 占比（太偏 = 近似常数 = 没信息）:")
    for name in ["HasPool", "Has2ndFlr", "HasBsmt", "HasGarage", "HasFireplace"]:
        v = FEATURE_BUILDERS[name](X)
        print(f"  {name:14s} 1 占 {v.mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
