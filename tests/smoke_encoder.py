"""验证类别编码（preprocess.build_encoder）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_encoder.py

要点：先填充（B1）再编码（B2）；只在 train 上 fit，test 只 transform。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw          # noqa: E402
from house_prices.preprocess import build_encoder, build_imputer  # noqa: E402


def main() -> None:
    train, test = load_raw()
    X, _y = get_xy(train)
    X_test = test.drop(columns=["Id"])

    # 先填充
    imp = build_imputer(X)
    X_imp = imp.fit_transform(X)
    X_test_imp = imp.transform(X_test)

    # 再编码（只在 train 上 fit）
    enc = build_encoder(X)
    X_enc = enc.fit_transform(X_imp)
    X_test_enc = enc.transform(X_test_imp)

    print(f"编码前: train {X_imp.shape}, test {X_test_imp.shape}")
    print(f"编码后: train {X_enc.shape}, test {X_test_enc.shape}")

    # 1) 是否全是数字、无 NaN
    non_num = X_enc.select_dtypes(exclude="number").columns.tolist()
    print(f"\n非数值列: {non_num}  (应为空)")
    print(f"NaN 个数: train {int(X_enc.isna().sum().sum())}, test {int(X_test_enc.isna().sum().sum())}")

    # 2) train / test 列是否完全一致（不一致会让预测错位）
    same = list(X_enc.columns) == list(X_test_enc.columns)
    print(f"train/test 列完全一致? {same}")

    # 3) 有序编码长什么样
    print("\n有序编码示例（ExterQual 的取值分布）:")
    print(X_enc["ExterQual"].value_counts().sort_index().to_string())
    print("  → 0=None 1=Po 2=Fa 3=TA 4=Gd 5=Ex")

    # 4) One-Hot 展开示例
    oh_cols = [c for c in X_enc.columns if c.startswith("Neighborhood_")]
    print(f"\nNeighborhood 展开成 {len(oh_cols)} 列，例如: {oh_cols[:3]}")


if __name__ == "__main__":
    main()
