"""阶段 E · E-3：让 CatBoost 用**原生类别特征**（`cat_features`），不做 One-Hot。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_catboost_native.py       # 完整 5×10 折（准，但慢）
    .\\.venv\\Scripts\\python.exe tests\\run_catboost_native.py 1     # 快速 5 折（先看趋势）

为什么给脚本加一个“重复次数”参数？
    完整 5×10 = 50 次训练，很容易变成“等十五分钟才发现方向不对”。
    先用 1（即 5 折）跑一遍看趋势，确认值得再跑完整的。

⚠️ 踩过的坑：不要随手加 `thread_count=-1`。
    小数据 + 多折场景下，“开满所有核心”会因为线程调度开销而**变得更慢**
    （实测卡到 1.8 小时 CPU 时间还没跑完一组）→ 不写这个参数，用库自己的默认策略。

为什么要试这个？
- 本项目 79 个特征里有 **43 个是类别列**。我们目前的做法是 **One-Hot**：
  43 列 → 两百多列 0/1，绝大多数格子里是 0（**又稀又大**），还丢掉了“类别 vs 目标”的关系。
- CatBoost 的看家本事就是**直接吃原始类别列**：它在内部用“有序目标统计量”
  （ordered target statistics，带折内防泄漏机制）把类别转成有用的数值。

⚠️ 所以这一步改的是**“表示方式”**，不是“调参”——属于更根本的改动。
  为了公平，**超参数与特征配方完全不动**（沿用 E-2 的当前最好配置）。

对照组设计（一次只改一个变量）：
  ① 现状：填充 → One-Hot/Ordinal → CatBoost   （= E-2 的最好配置，分数应复现 0.11993）
  ② 原生：填充 →（不编码）→ CatBoost(cat_features=43 列)
  ③ 追加：在 ② 基础上，把 MSSubClass 也**当类别列**（它是“数字形式的类别”，见工程规范）

⚠️ 第 ③ 组的坑：MSSubClass 原本是 int64，经过数值列的 `SimpleImputer` 后会被转成
   **float（60.0）**，而 CatBoost 只接受 int / str：
   `CatBoostError: Invalid type for cat_feature[...]=60.0 : cat_features must be
    integer or string, real number values and NaN values should be converted to string.`
   → 所以要先 `.astype(str)` 把“数字形式的类别”变成真正的字符串类别。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw                    # noqa: E402
from house_prices.evaluate import cv_rmse_log                      # noqa: E402
from house_prices.preprocess import cat_feature_names, make_pipeline  # noqa: E402

BASE = ("QualArea", "TotalSF")
BEST = dict(iterations=1200, learning_rate=0.025, depth=6, random_seed=42)

# 命令行第 1 个参数 = 重复次数（不写就是 10）。sys.argv[0] 是脚本名，所以看 [1]。
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def model_with_cats(cat_cols: list[str] | None) -> CatBoostRegressor:
    """cat_cols=None → 走 One-Hot 管道，模型不需要知道类别列。

    ⚠️ 坑：`cat_features` 必须传**元组**，不能传列表！
       CatBoost 的构造西数会改写 list（转成内部形式），而 sklearn 的 clone
       要求“构造西数不得修改传入的参数”（身份检查）→ 传 list 会报
       `RuntimeError: Cannot clone object ... modifies parameter cat_features`。
       详细可复现探测：`tests/smoke_catboost_native.py` 第 ⑤ 项。
    """
    kwargs = dict(
        **BEST,
        verbose=0,
        allow_writing_files=False,
        # ⚠️ 不要加 thread_count=-1：小数据上开满线程反而更慢（见文件头说明）
    )
    if cat_cols:
        kwargs["cat_features"] = tuple(cat_cols)   # ⚠️ 元组，不是列表
    return CatBoostRegressor(**kwargs)


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    cats = cat_feature_names(X)
    # ③ 专用副本：把 MSSubClass 转成字符串类别（否则 CatBoost 报 float 类型错误）
    X_cat = X.assign(MSSubClass=X["MSSubClass"].astype(str))

    print(f"特征固定 {BASE}；超参数固定 {BEST}")
    print(f"类别列共 {len(cats)} 个（第 ③ 组再加上 MSSubClass）")
    print(f"CV：5 折 × {REPEATS} 次重复\n")

    cases = [
        ("① One-Hot 管道（现状·基准）", None, X),
        ("② 原生 cat_features（43 列）", cats, X),
        ("③ 原生 + MSSubClass 也算类别", cats + ["MSSubClass"], X_cat),
    ]

    for name, cat_cols, X_use in cases:
        model = model_with_cats(cat_cols)
        pipe = make_pipeline(
            model,
            X_use,
            derived_features=BASE,
            encode=cat_cols is None,     # ① 编码；②③ 不编码（交给 CatBoost）
        )
        t0 = time.perf_counter()
        mean, std = cv_rmse_log(pipe, X_use, y, n_repeats=REPEATS)
        used = time.perf_counter() - t0
        print(f"{name}  →  {mean:.5f} ± {std:.5f}   （用时 {used:.1f}s）")


if __name__ == "__main__":
    main()
