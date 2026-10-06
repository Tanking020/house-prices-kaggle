"""阶段 D · D-3 实验：换用三个专业梯度提升库（XGBoost / LightGBM / CatBoost）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_boosting_experiment.py

前置：三个库已安装（见 requirements.txt）。若报 ModuleNotFoundError，先跑：
    .\\.venv\\Scripts\\python.exe -m pip install xgboost lightgbm catboost ^
        -i https://pypi.tuna.tsinghua.edu.cn/simple

对照设计（只换【库】这一个变量）：
- 树数统一 300、学习率统一 0.05、随机种子统一 42，其余参数用各自默认值；
- 特征固定 ("QualArea", "TotalSF")，预处理管道不变；
- 第 ① 行是上轮 sklearn GBDT(300, lr=0.05) 的"复现锚点"：
  分数与 #12 接近 → 说明这次对比是公平的（同一把尺子）。

⚠️ 三个库的参数名并不统一，这是最容易踩的坑：
    "树数"    sklearn = n_estimators   LightGBM/XGBoost = n_estimators   CatBoost = iterations
    "随机种子" sklearn = random_state   CatBoost = random_seed
"""
from __future__ import annotations

import sys
from pathlib import Path

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw        # noqa: E402
from house_prices.evaluate import cv_rmse_log         # noqa: E402
from house_prices.preprocess import make_pipeline     # noqa: E402


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    # 特征固定为"目前最好的一组"，只变模型 → 逐行可归因
    BASE = ("QualArea", "TotalSF")

    cases = {
        "① sklearn GBDT(300, lr=0.05)【上轮赢家·锚点】": (
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, random_state=42
            ),
            BASE,
        ),
        "② XGBoost(300, lr=0.05)": (
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            BASE,
        ),
        "③ LightGBM(300, lr=0.05)": (
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,          # -1 = 不打印训练日志（否则刷屏）
            ),
            BASE,
        ),
        "④ CatBoost(300, lr=0.05)": (
            CatBoostRegressor(
                iterations=300,      # ⚠️ CatBoost 里"树数"叫 iterations
                learning_rate=0.05,
                random_seed=42,      # ⚠️ 这里叫 random_seed，不叫 random_state
                verbose=0,           # 0 = 不打印训练日志
                allow_writing_files=False,   # 不生成 catboost_info/ 日志目录（别污染仓库）
            ),
            BASE,
        ),
    }

    print(f"特征固定：{BASE}\nCV：5×10 折重复交叉验证（RMSE in log space）\n")
    print("【第一轮】树数/学习率/种子统一，其余用各自默认值")
    for name, (model, feats) in cases.items():
        pipe = make_pipeline(model, X, derived_features=feats)
        mean, std = cv_rmse_log(pipe, X, y, n_repeats=10)
        print(f"{name}  →  {mean:.5f} ± {std:.5f}")

    # ---- 第二轮：诊断"为什么 XGBoost / LightGBM 反而更差" ----
    # 假设：① 之所以赢，是因为 sklearn GBDT 默认 max_depth=3（浅树、不易过拟合）；
    #       而 XGBoost 默认 max_depth=6、LightGBM 默认 num_leaves=31（≈5 层），
    #       在 1168 行训练集上太复杂 → 过拟合。
    # 验证方法：只把"树复杂度"压到 3 层，其它一律不动，看分数是否追回来。
    #   → 若追回：说明差在树复杂度，而不是"这个库不行"；后续调参重点就是 depth/leaves。
    #   → 若追不回：说明假设错误，需要另找原因。
    depth_cases = {
        "⑤ XGBoost(300, lr=0.05, max_depth=3)": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "⑥ LightGBM(300, lr=0.05, max_depth=3, num_leaves=8)": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,             # 限制"最多几层"
            num_leaves=8,            # LightGBM 是叶子生长，必须同时限制"最多几片叶子"
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    print("\n【第二轮】只压树复杂度到 3 层（诊断上一步的『更差』从哪来）")
    for name, model in depth_cases.items():
        pipe = make_pipeline(model, X, derived_features=BASE)
        mean, std = cv_rmse_log(pipe, X, y, n_repeats=10)
        print(f"{name}  →  {mean:.5f} ± {std:.5f}")


if __name__ == "__main__":
    main()
