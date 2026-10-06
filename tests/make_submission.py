"""生成 Kaggle 提交文件：全量 train 训练 → 预测 test → expm1 还原 → 格式自检 → 写出。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\make_submission.py        # 默认 v2
    .\\.venv\\Scripts\\python.exe tests\\make_submission.py v3     # 指定版本号

产出：submissions/submission_<版本号>.csv（列名 Id,SalePrice；1459 行 + 表头）

模型来源：`src/house_prices/models.py` 的 `make_best_model()`
（**单一事实来源** —— 改模型只改那一个文件，不用在提交脚本里再拄一遍参数）。

历史：
- v1（已提交，LB 0.14206）：`Ridge(alpha=1)` + 无派生特征
- v2：当前最好 —— `CatBoost(lr=0.025, iters=1200, depth=6)` + QualArea + TotalSF（CV 0.11993）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import RAW_DIR, get_xy, load_raw            # noqa: E402
from house_prices.models import (                                   # noqa: E402
    BEST_DERIVED_FEATURES,
    BEST_TARGET_ENCODE,
    make_best_model,
)
from house_prices.preprocess import make_pipeline                  # noqa: E402

# 命令行第 1 个参数 = 版本号（不写就是 v2）；sys.argv[0] 是脚本名
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
OUT_PATH = ROOT / "submissions" / f"submission_{VERSION}.csv"


def main() -> None:
    train, test = load_raw()
    X, y = get_xy(train)

    # ---- 1) 用【全量】train 训练（提交时不再留验证集）----
    # 模型与特征配方都从 src/house_prices/models.py 取（单一事实来源）
    pipe = make_pipeline(
        make_best_model(), X,
        derived_features=BEST_DERIVED_FEATURES,
        target_encode=BEST_TARGET_ENCODE,
    )
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
    print(f"预测价格范围: {sample['SalePrice'].min():,.0f} ~ {sample['SalePrice'].max():,.0f}")
    # 对照训练集范围：树模型不能外推，预测值应落在训练范围内（基线 Ridge 曾冲到 1,071,406 → 明显外推过头）
    print(f"训练集范围  : {y.min():,.0f} ~ {y.max():,.0f}")
    print("\n前 5 行：")
    print(sample.head().to_string(index=False))
    print("\n前 5 行：")
    print(sample.head().to_string(index=False))


if __name__ == "__main__":
    main()
