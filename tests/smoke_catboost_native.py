"""冒烟：验证"CatBoost 原生类别特征"这条路走得通（**不算分数**）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_catboost_native.py

为什么需要它？
- CatBoost 的 `cat_features` 要求这些列是 **int / str**，不能是 float 的 NaN；
  而 pandas 3.0 的文本列 dtype 是 `str`（不是过去的 `object`）→ 版本差异容易踩坑。
- 直接跑 12 分钟的完整实验前，先用 100 行数据把"能不能跑通"验证掉。

四项体检：
  ① 类别列数对不对（应为 43）
  ② 填充后还有没有 NaN
  ③ `cat_features=[...]` + 跳过编码的流水线能否 fit / predict
  ④ 预测值是不是有限数（不是 NaN / inf）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw                        # noqa: E402
from house_prices.preprocess import (                                  # noqa: E402
    build_preprocessor,
    cat_feature_names,
    make_pipeline,
)

BASE = ("QualArea", "TotalSF")


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    # 只取 200 行 + 2 棵树，速度优先（只看"通不通"，不看效果）
    Xs, ys = X.iloc[:200].copy(), y.iloc[:200].copy()

    cats = cat_feature_names(Xs)
    print(f"① 类别列数: {len(cats)}   (应为 43)")

    # ② 用与实验完全相同的预处理（只是不编码），先看填充结果
    prep = build_preprocessor(Xs, derived_features=BASE, encode=False)
    Xt = prep.fit_transform(Xs)
    print(f"② {len(BASE)} 个派生特征已加入 → 形状 {Xt.shape}")
    print(f"   填充后 NaN 个数: {int(Xt.isna().sum().sum())}   (应为 0)")
    print(f"   类别列 dtype 示例: {Xt[cats[0]].dtype}")

    # ③ 真正的流水线：填充 →（不编码）→ CatBoost(cat_features=...)
    model = CatBoostRegressor(
        iterations=2, depth=3, verbose=0, allow_writing_files=False,
        cat_features=cats,
    )
    pipe = make_pipeline(model, Xs, derived_features=BASE, encode=False)
    pipe.fit(Xs, np.log1p(ys))
    pred = pipe.predict(Xs)
    print(f"③ fit / predict 成功，预测形状 {pred.shape}")

    # ④ 预测值必须是有限数
    ok = bool(np.all(np.isfinite(pred)))
    print(f"④ 预测值全部有限（非 NaN/inf）? {ok}")
    print(f"   预测前 5 个: {np.round(pred[:5], 3)}")
    print(f"   真实前 5 个: {np.round(np.log1p(ys[:5]), 3)}")

    if not ok:
        raise SystemExit("❌ 预测值里有 NaN/inf，不能用于实验")

    # ⑤ clone 兼容性探测
    # ⚠️ 坑：cv_rmse_log 每折都要 clone(model)，而 CatBoost 的 __init__ 会"改写"
    #    cat_features 参数 → sklearn.clone 的身份检查失败：
    #    RuntimeError: Cannot clone object ..., as the constructor either does not
    #    set or modifies parameter cat_features
    # 这里把几种传参方式都试一遍，找出"能过 clone"的写法。
    print("\n⑤ clone 兼容性探测（cv_rmse_log 每折都要 clone）")
    variants = [
        ("① 列表列名 list[str]", {"cat_features": cats}),
        ("② 元组列名 tuple[str, ...]", {"cat_features": tuple(cats)}),
        ("③ 列表下标 list[int]", {"cat_features": list(range(len(cats)))}),
        ("④ 元组下标 tuple[int, ...]", {"cat_features": tuple(range(len(cats)))}),
    ]
    for label, kw in variants:
        m = CatBoostRegressor(
            iterations=2, depth=3, verbose=0,
            allow_writing_files=False, **kw,
        )
        try:
            clone(m)
            print(f"   {label:28s} → ✅ 可以 clone")
        except Exception as exc:      # noqa: BLE001 - 这里就是要捕获所有异常看现象
            print(f"   {label:28s} → ❌ {type(exc).__name__}: {str(exc)[:48]}")


if __name__ == "__main__":
    main()
