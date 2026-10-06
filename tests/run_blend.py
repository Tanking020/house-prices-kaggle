"""阶段 F · F-1：加权平均（blending）—— 把多个模型的预测按权重"拼"起来。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_blend.py            # 默认池 A
    .\\.venv\\Scripts\\python.exe tests\\run_blend.py 0.05 B     # 步长 0.05 + 池 B

为什么融合可能有用？（核心直觉）
- 假设模型 A 在某套房上猜高 5%、模型 B 在**同一套**房上猜低 3%，
  平均后误差互相抵消 → 比任何单个模型都准。
- ⭐ **收益来自"误差不相关"**：如果两个模型犯的错一模一样，平均毫无用处。
  所以本脚本第 ③ 步专门打印**误差的相关系数矩阵**。

⭐ 关键方法论：怎样公平地评估"加权后的分数"？
- ❌ 直接用 CV：同一批数据既用来**调权重**又用来**报分数** → 分数会偏乐观（选择偏差）。
- ✅ 用 **OOF 预测**（out-of-fold）：把训练集切成 K 折，
  每折用"其余折"训练后预测**本折** → 每一行都拿到一个"没见过它"的预测值。
  之后**任何**权重组合都可以在这份 OOF 预测上直接打分，一行代码、零泄漏。

⚠️ 但注意：在 OOF 上**挑最优权重**仍然是一次"选择偏差"（从 200+ 个组合里选最小）。
   所以最后会额外打印"前 5 名"，让你看到**最优附近是一片平台**（差不了多少），
   别把第 1 名当成"精确答案"。
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw          # noqa: E402
from house_prices.evaluate import rmse                   # noqa: E402
from house_prices.models import make_best_model          # noqa: E402
from house_prices.preprocess import make_pipeline        # noqa: E402

BASE = ("QualArea", "TotalSF")
SEED = 42
N_SPLITS = 5

# 权重网格的步长（命令行第 1 个参数）：0.05 → 每个权重取 0, 0.05, ..., 1
STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

# 所有候选基模型（名字 → 造模型的函数）
# 注意：融合池要刻意选**差异大**的模型（线性 / Bagging / Boosting）
CANDIDATES = {
    "Ridge(alpha=1)": lambda: Ridge(alpha=1.0),
    "随机森林(200)": lambda: RandomForestRegressor(
        n_estimators=200, random_state=SEED, n_jobs=-1
    ),
    "GBDT(300,lr0.05)": lambda: GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, random_state=SEED
    ),
    "XGBoost(depth3)": lambda: XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        random_state=SEED, n_jobs=-1, verbosity=0,
    ),
    "CatBoost(最优配置)": make_best_model,
}

# 融合池：用哪些基模型一起掺——这是 F-1 最关键的**实验设计选择**
#   池 A：弱强混合——故意把“差很多”的 Ridge / 随机森林 和最好的 CatBoost 放一起，看会怎样
#   池 B：强度接近——三个都是 Boosting 家族、CV 都在 0.120~0.126，看误差是否更互补
POOLS = {
    "A": ["Ridge(alpha=1)", "随机森林(200)", "CatBoost(最优配置)"],
    "B": ["GBDT(300,lr0.05)", "XGBoost(depth3)", "CatBoost(最优配置)"],
}
POOL = (sys.argv[2] if len(sys.argv) > 2 else "A").upper()


def oof_predictions(make_model, X, y_log) -> np.ndarray:
    """生成 OOF 预测：每折用其余折训练，只预测本折 → 每行恰好被预测一次。"""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_log))
    for tr_idx, va_idx in kf.split(X):
        pipe = make_pipeline(make_model(), X, derived_features=BASE)
        pipe.fit(X.iloc[tr_idx], y_log[tr_idx])
        oof[va_idx] = pipe.predict(X.iloc[va_idx])
    return oof


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)
    y_log = np.log1p(y).to_numpy()      # 统一成 numpy 数组，方便按整数下标取值

    # ---- ① 为池中各模型生成 OOF 预测 ----
    names = POOLS[POOL]
    print(f"融合池 {POOL}：{names}")
    print(f"特征固定 {BASE}；OOF 用 {N_SPLITS} 折（seed={SEED}）\n")
    oofs: dict[str, np.ndarray] = {}
    singles: dict[str, float] = {}
    for name in names:
        make_model = CANDIDATES[name]
        t0 = time.perf_counter()
        oof = oof_predictions(make_model, X, y_log)
        used = time.perf_counter() - t0
        oofs[name] = oof
        singles[name] = rmse(y_log, oof)
        print(f"  生成 OOF: {name:20s} （用时 {used:5.1f}s）")

    # ---- ② 单模型 OOF 成绩（顺便当"锚点"校验） ----
    print("\n① 单模型 OOF 成绩（对比历史 CV，验证流程没写错）")
    for name, s in sorted(singles.items(), key=lambda kv: kv[1]):
        print(f"  {name:20s} OOF RMSE(log) = {s:.5f}")

    # ---- ③ 误差相关性：融合收益的来源 ----
    print("\n② 误差相关系数矩阵（越接近 0 越「互补」；接近 1 说明犯的错一样，平均没用）")
    errs = np.column_stack([oofs[n] - y_log for n in names])
    corr = np.corrcoef(errs, rowvar=False)
    header = "           " + "".join(f"{n[:10]:>12s}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = "".join(f"{corr[i, j]:>12.3f}" for j in range(len(names)))
        print(f"{n[:10]:>10s} {row}")

    # ---- ④ 等权平均（最省事的融合） ----
    equal = np.mean([oofs[n] for n in names], axis=0)
    best_single_name = min(singles, key=lambda k: singles[k])
    print(f"\n③ 等权平均（各 {1/len(names):.3f}）")
    print(f"  OOF RMSE(log) = {rmse(y_log, equal):.5f}"
          f"   （最好的单模型 {best_single_name} = {singles[best_single_name]:.5f}）")

    # ---- ⑤ 网格搜索权重（权重和为 1） ----
    # itertools.product(A, B) = A 的每个元素与 B 的每个元素配对（笛卡尔积）
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, STEP), 6)
    results: list[tuple[float, tuple[float, ...]]] = []
    for head in itertools.product(grid, repeat=len(names) - 1):
        last = 1.0 - sum(head)
        if last < -1e-9 or last > 1.0 + 1e-9:
            continue          # 最后一个权重必须落在 [0, 1] 内
        w = tuple(head) + (max(last, 0.0),)
        pred = sum(wi * oofs[n] for wi, n in zip(w, names))
        results.append((rmse(y_log, pred), w))
    results.sort(key=lambda t: t[0])

    print(f"\n④ 权重网格搜索（步长 {STEP}，共试了 {len(results)} 组）—— 前 5 名")
    print("   排名 | OOF RMSE(log) | " + " | ".join(f"{n[:10]:>10s}" for n in names))
    for rank, (s, w) in enumerate(results[:5], start=1):
        cells = " | ".join(f"{wi:>10.2f}" for wi in w)
        print(f"   {rank:>4} | {s:>13.5f} | {cells}")

    best_score, best_w = results[0]
    print(f"\n   → 最优权重 {dict(zip(names, [round(w, 3) for w in best_w]))}")
    print(f"   → 相比最好单模型（{singles[best_single_name]:.5f}）"
          f"改善 {singles[best_single_name] - best_score:+.5f}")


if __name__ == "__main__":
    main()
