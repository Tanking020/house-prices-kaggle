"""验证缺失值处理（preprocess.build_imputer）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_preprocess.py

要点：
- 只在 train 上 fit，test 只 transform（防泄漏）
- 填充后 train / test 都不应再有 NaN
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw          # noqa: E402
from house_prices.preprocess import build_imputer        # noqa: E402


def main() -> None:
    train, test = load_raw()
    X, _y = get_xy(train)
    X_test = test.drop(columns=["Id"])

    print(f"原始 NaN → train: {int(X.isna().sum().sum())}, test: {int(X_test.isna().sum().sum())}")

    imp = build_imputer(X)
    X_imp = imp.fit_transform(X)        # 只在 train 上 fit
    X_test_imp = imp.transform(X_test)  # test 只 transform

    print(f"填充后 NaN → train: {int(X_imp.isna().sum().sum())}, test: {int(X_test_imp.isna().sum().sum())}")
    print(f"形状 → train {X_imp.shape}, test {X_test_imp.shape}")

    # 抽几个关键列看看填成了什么
    print("\n抽样检查：")
    print("  MasVnrType 取值:", sorted(X_imp["MasVnrType"].unique())[:6])
    print("  GarageYrBlt 最小值:", X_imp["GarageYrBlt"].min())
    print("  LotFrontage 里的 NaN 被填成了:", X_imp["LotFrontage"].isna().sum(), "个 NaN")


if __name__ == "__main__":
    main()
