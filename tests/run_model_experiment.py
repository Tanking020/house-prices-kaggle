"""阶段 D · 模型升级实验：只换【模型】这一个变量（预处理与特征保持不变）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_model_experiment.py

说明：
- 本阶段先用 sklearn 自带的模型（无需安装新包）：随机森林 / GBDT。
- GBDT 是"串行"训练（每棵树纠正前面的残差），比随机森林慢一些。
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

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
        "① Ridge(alpha=1)【线性参考】": (Ridge(alpha=1.0), BASE),
        "② 随机森林(200 棵)【上轮赢家】": (
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            BASE,
        ),
        "③ GBDT（默认 100 棵, lr=0.1）": (
            GradientBoostingRegressor(random_state=42),
            BASE,
        ),
        "④ GBDT（300 棵, lr=0.05）": (
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, random_state=42
            ),
            BASE,
        ),
    }

    print("模型                              CV (RMSE-log, 5×10)")
    print("-" * 62)
    for name, (model, feats) in cases.items():
        pipe = make_pipeline(model, X, derived_features=feats)
        mean, std = cv_rmse_log(pipe, X, y, n_repeats=10)
        print(f"{name:32s} {mean:.5f} ± {std:.5f}")


if __name__ == "__main__":
    main()
