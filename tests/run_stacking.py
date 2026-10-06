"""阶段 F · F-2：Stacking —— 让"元模型"自己学怎么组合基模型。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_stacking.py

和 F-1（加权平均）的区别：
- 加权平均：权重是**人定**的（等权 / 网格搜出来）。
- Stacking：把各基模型的预测当成**新特征**，再训一个"元模型"（meta-model）来**自己学**组合方式。
  ⭐ 元模型还能学"哪个模型更可靠" —— 本质上是"用学习代替手调权重"。

⚠️ 本脚本为什么比常见写法多一层循环？（**这是 Stacking 唯一的难点**）
- 元特征必须用 **OOF 生成**：否则元模型看到的是"基模型在训练集上的答案"→ 严重泄漏。
- 但只用"一层 OOF + 在 OOF 上做 CV"仍有残留乐观（元特征是在**全部**训练行上生成的）。
- 所以这里用 **嵌套 CV**：
    外层 5 折 —— 只用来**评估** stacking 的分数（每折的验证行对全程都不可见）；
    内层 5 折 —— 只在**外层训练行内部**生成元特征，用来训元模型。
  代价是训练次数多（外层 5 × (内层 5 × 3 模型 + 3 模型) = 90 次），换来的是**干净的分数**。

池子（沿用 F-1 的池 B，三个实力接近的 Boosting）：
    GBDT(300, lr=0.05) / XGBoost(depth=3) / CatBoost(最优配置)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw        # noqa: E402
from house_prices.evaluate import rmse                # noqa: E402
from house_prices.models import make_best_model       # noqa: E402
from house_prices.preprocess import make_pipeline     # noqa: E402

BASE = ("QualArea", "TotalSF")
SEED = 42
N_SPLITS = 5

MODELS = {
    "GBDT(300,lr0.05)": lambda: GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, random_state=SEED
    ),
    "XGBoost(depth3)": lambda: XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        random_state=SEED, n_jobs=-1, verbosity=0,
    ),
    "CatBoost(最优配置)": make_best_model,
}


def fit_predict(make_model, X_tr, y_tr, X_te) -> np.ndarray:
    """在训练集上训练一条完整流水线，然后预测。"""
    pipe = make_pipeline(make_model(), X_tr, derived_features=BASE)
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def simple_oof(make_model, X, y_log) -> np.ndarray:
    """普通 5 折 OOF（只为打印单模型成绩做参照，不做元特征）。"""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_log))
    for tr, va in kf.split(X):
        oof[va] = fit_predict(make_model, X.iloc[tr], y_log[tr], X.iloc[va])
    return oof


def stacking_nested(X, y_log) -> tuple[np.ndarray, np.ndarray]:
    """嵌套 CV 评估 Stacking。返回 (外层 OOF 预测, 元模型系数的平均值)。"""
    outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(len(y_log))
    coefs = []

    for fold, (tr, va) in enumerate(outer.split(X), start=1):
        # ---- 内层：只在外层训练行内部，用 K 折生成元特征 ----
        inner = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        meta_tr = np.zeros((len(tr), len(MODELS)))
        for i_tr, i_va in inner.split(tr):
            X_in, y_in = X.iloc[tr[i_tr]], y_log[tr[i_tr]]
            X_hold = X.iloc[tr[i_va]]
            for j, make_model in enumerate(MODELS.values()):
                meta_tr[i_va, j] = fit_predict(make_model, X_in, y_in, X_hold)

        # ---- 元模型：用内层元特征训练 ----
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_tr, y_log[tr])
        coefs.append(meta_model.coef_)

        # ---- 外层验证折的元特征：基模型在【外层训练行全体】上重训后预测 ----
        meta_va = np.zeros((len(va), len(MODELS)))
        for j, make_model in enumerate(MODELS.values()):
            meta_va[:, j] = fit_predict(make_model, X.iloc[tr], y_log[tr], X.iloc[va])

        oof_meta[va] = meta_model.predict(meta_va)
        print(f"    外层第 {fold} 折完成")

    return oof_meta, np.mean(coefs, axis=0)


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)
    y_log = np.log1p(y).to_numpy()

    print(f"特征固定 {BASE}；嵌套 CV（外层 {N_SPLITS} 折 / 内层 {N_SPLITS} 折）\n")

    # ---- ① 单模型 OOF 成绩（参照） ----
    print("① 单模型 OOF 成绩（参照）")
    singles = {}
    oofs_single = {}
    for name, make_model in MODELS.items():
        t0 = time.perf_counter()
        oof = simple_oof(make_model, X, y_log)
        oofs_single[name] = oof
        singles[name] = rmse(y_log, oof)
        print(f"  {name:20s} {singles[name]:.5f}   （用时 {time.perf_counter() - t0:.1f}s）")

    best_single = min(singles.values())
    print(f"\n  最好的单模型 = {best_single:.5f}")

    # ---- ② Stacking（嵌套 CV） ----
    print("\n② Stacking（嵌套 CV，正在跑，约 3~5 分钟）")
    t0 = time.perf_counter()
    oof_meta, coefs = stacking_nested(X, y_log)
    used = time.perf_counter() - t0
    stack_score = rmse(y_log, oof_meta)

    print(f"\n  Stacking OOF RMSE(log) = {stack_score:.5f}   （用时 {used:.1f}s）")
    print(f"  相比最好单模型改善 {best_single - stack_score:+.5f}")

    # ---- ③ 元模型学到了什么权重？ ----
    print("\n③ 元模型学到的权重（5 折平均；可正可负，正值 = 被重用）")
    for name, c in zip(MODELS, coefs):
        print(f"  {name:20s} 权重 = {c:+.4f}")
    print(f"  （权重之和 = {coefs.sum():.4f}，接近 1 说明主要还是在做'加权平均'）")

    # ---- ④ 对照：实践中常用的"标准做法"（单层 OOF） ----
    # 直接拿①里已经算好的 OOF 当元特征，再对元模型做一次普通 5 折 CV。
    # ⚠️ 它**偏乐观**：元特征是在**全部**训练行上生成的（含 CV 的验证行），
    #    而嵌套 CV（②）**偏严格**（元模型是在更弱的"内层元特征"上训的）。
    #    两个数夹逼一下，就知道 Stacking 到底值不值。
    Z = np.column_stack([oofs_single[n] for n in MODELS])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pred_std = np.zeros(len(y_log))
    for tr, va in kf.split(Z):
        meta = Ridge(alpha=1.0).fit(Z[tr], y_log[tr])
        pred_std[va] = meta.predict(Z[va])
    std_score = rmse(y_log, pred_std)

    print("\n④ 对照：标准做法（单层 OOF）的估计")
    print(f"  标准做法（理论上偏乐观） = {std_score:.5f}   （相对最好单模型 {best_single - std_score:+.5f}）")
    print(f"  嵌套 CV（理论上偏严格）   = {stack_score:.5f}   （相对最好单模型 {best_single - stack_score:+.5f}）")
    print("  → 两种口径都 ≈ 0.126，**均差于**最好单模型 → Stacking 在本项目是负收益")


if __name__ == "__main__":
    main()
