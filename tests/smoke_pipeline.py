"""验证完整 Pipeline（填充 → 编码 → 模型）能端到端跑通。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_pipeline.py

注意：这里只验证"能跑通"，**不算分数**（CV 分数属于 B4）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw              # noqa: E402
from house_prices.preprocess import build_preprocessor, make_pipeline  # noqa: E402


def main() -> None:
    train, test = load_raw()
    X, y = get_xy(train)
    X_test = test.drop(columns=["Id"])

    # ---- 1) 预处理流水线（填充 → 编码）----
    prep = build_preprocessor(X)
    Xt = prep.fit_transform(X)          # 只在 train 上 fit
    Xt_test = prep.transform(X_test)    # test 只 transform

    print(f"预处理输出: train {Xt.shape}, test {Xt_test.shape}")
    print(f"NaN 个数: train {int(Xt.isna().sum().sum())}, test {int(Xt_test.isna().sum().sum())}")
    print(f"输出列数（get_feature_names_out）: {len(prep.get_feature_names_out())}")

    # ---- 2) 整条流水线（含模型）端到端 ----
    pipe = make_pipeline(Ridge(), X)
    pipe.fit(X, np.log1p(y))             # 训练目标在 log 空间
    pred_log = pipe.predict(X_test)      # 模型输出也在 log 空间
    prices = np.expm1(pred_log)          # 还原回原始价格尺度

    print(f"\n整条流水线预测完成: 形状 {pred_log.shape}")
    print(f"train 房价范围: {y.min():,.0f} ~ {y.max():,.0f}")
    print(f"预测房价范围:   {prices.min():,.0f} ~ {prices.max():,.0f}  (均值 {prices.mean():,.0f})")

    # ---- 3) Pipeline 的两个实用操作 ----
    print(f"\n用 named_steps 取模型: {type(pipe.named_steps['model']).__name__}")
    pipe.set_params(model__alpha=10.0)
    print(f"用 set_params 改参数 model__alpha -> {pipe.named_steps['model'].alpha}")


if __name__ == "__main__":
    main()
