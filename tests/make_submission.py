"""B5 · 首次提交：全量 train 训练 → 预测 test → 写出 submission。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\make_submission.py

产出：submissions/submission_v1.csv（列名 Id,SalePrice；1459 行 + 表头）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import RAW_DIR, get_xy, load_raw   # noqa: E402
from house_prices.preprocess import make_pipeline         # noqa: E402

OUT_PATH = ROOT / "submissions" / "submission_v1.csv"


def main() -> None:
    train, test = load_raw()
    X, y = get_xy(train)

    # ---- 1) 用【全量】train 训练（提交时不再留验证集）----
    pipe = make_pipeline(Ridge(alpha=1.0), X)
    pipe.fit(X, np.log1p(y))                 # 目标在 log 空间训练

    # ---- 2) 预测 test，并 expm1 还原回原始价格尺度 ----
    pred_log = pipe.predict(test.drop(columns=["Id"]))
    prices = np.expm1(pred_log)

    # ---- 3) 按 sample_submission 的 Id 顺序对齐 ----
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    pred_map = pd.Series(prices, index=test["Id"])
    sample["SalePrice"] = sample["Id"].map(pred_map)     # 按 Id 对齐，避免错位

    # ---- 4) 格式自检（提交格式错 = 直接失败）----
    sample_ids = pd.read_csv(RAW_DIR / "sample_submission.csv")["Id"].tolist()
    checks = {
        "列名 == ['Id', 'SalePrice']": list(sample.columns) == ["Id", "SalePrice"],
        "行数 == 1459": len(sample) == 1459,
        "Id 最小/最大 == 1461 / 2919": (sample["Id"].min(), sample["Id"].max()) == (1461, 2919),
        "Id 顺序与 sample_submission 完全一致": sample["Id"].tolist() == sample_ids,
        "无缺失值": bool(not sample.isna().any().any()),
        "预测值全为正": bool((sample["SalePrice"] > 0).all()),
    }
    print("格式自检：")
    for name, ok in checks.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    if not all(checks.values()):
        raise SystemExit("\n格式自检未通过 → 不写出文件")

    # ---- 5) 写出（不带 pandas 索引列）----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)

    print(f"\n已写出: {OUT_PATH.relative_to(ROOT)}")
    print(f"价格范围: {sample['SalePrice'].min():,.0f} ~ {sample['SalePrice'].max():,.0f}")
    print("\n前 5 行：")
    print(sample.head().to_string(index=False))


if __name__ == "__main__":
    main()
