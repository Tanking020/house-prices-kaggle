"""阶段 E · E-2 复核：换一套数据划分，检验"表 A 挑出来的最优"是不是碰运气。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\run_tuning_seed_check.py

为什么需要这个脚本？（一个很容易被忽略的方法论陷阱）
- `run_tuning_lr_iters.py` 的表 A 扫了 9 个配置，然后**取了分数最低的那个**。
- 但"从 N 个候选里挑最小值"这个动作**本身就会让分数偏乐观** ——
  相当于同一份卷子考 13 次、取最高分当成绩（选择偏差 / selection bias）。
- 更麻烦的是：我们始终用的是**同一套**折（seed=42），所以"最优"可能是**这套折的巧合**。

复核办法（本脚本）：
  把同一批配置**换一套数据划分**（seed 42 → 2024）再跑一遍，比较**排名**：
    排名基本不变 → 结论稳健，可以放心采用；
    排名大幅翻盘 → 原来的"最优"只是碰运气，别写进实验结论。

⚠️ 注意区分两种"换种子"：
  ① 换**数据划分**（本脚本）：考的是"结论换一批题目还成立吗" → 这是我们要的
  ② 换**模型自身的随机种子**：考的是"模型训练稳不稳" → 不是本脚本的目的
"""
from __future__ import annotations

import sys
from pathlib import Path

from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.data import get_xy, load_raw        # noqa: E402
from house_prices.evaluate import cv_rmse_log         # noqa: E402
from house_prices.preprocess import make_pipeline     # noqa: E402

BASE = ("QualArea", "TotalSF")

# 表 A 的**前三名** + E-1 的默认配置（旧最好，作参照）
# 用"元组列表"而不是字典：这里要保留顺序，而且每项有两层含义
CANDIDATES = [
    (0.025, 1200),   # 表 A 第 1 名（0.11993）
    (0.05, 1200),    # 表 A 第 2 名（0.12006）
    (0.05, 600),     # 表 A 第 3 名（0.12058）
    (0.05, 300),     # E-1 默认配置（0.12263）—— 旧最好，看新配置是否真的赢它
]
SEEDS = [42, 2024]


def score(X, y, lr: float, iters: int, seed: int) -> float:
    """固定 depth=6、特征不变，用指定 seed 的 5×10 折打分。"""
    model = CatBoostRegressor(
        iterations=iters,
        learning_rate=lr,
        depth=6,
        random_seed=42,              # 模型自身的随机种子固定 → 只让"数据划分"变
        verbose=0,
        allow_writing_files=False,
    )
    mean, _std = cv_rmse_log(
        make_pipeline(model, X, derived_features=BASE), X, y, n_repeats=10, seed=seed
    )
    return mean


def main() -> None:
    train, _test = load_raw()
    X, y = get_xy(train)

    print("换数据划分复核（每格 = CV 均值；两个 seed 的折划分完全不同）\n")
    head = "配置 (lr, iterations) " + "".join(f"| seed={s:<7}" for s in SEEDS)
    print(head)
    print("-" * (len(head) + 12))

    table: dict[tuple[float, int], dict[int, float]] = {}
    for lr, it in CANDIDATES:
        table[(lr, it)] = {}
        for s in SEEDS:
            table[(lr, it)][s] = score(X, y, lr, it, seed=s)
        row = "".join(f"| {table[(lr, it)][s]:.5f}  " for s in SEEDS)
        print(f"lr={lr:<5g} iters={it:<5}   {row}")

    # 每个 seed 下各自排名，看顺序是否一致
    print("\n各 seed 下的排名（1 = 该划分下最好）：")
    for s in SEEDS:
        order = sorted(CANDIDATES, key=lambda c: table[c][s])
        names = [f"({lr:g},{it})" for lr, it in order]
        print(f"  seed={s:<5}: " + "  >  ".join(names))

    # 附：两个 seed 的"新旧最好"差值，判断改进是否大于种子波动
    print("\n对照（新最好 vs 旧最好）：")
    for s in SEEDS:
        new_best = min(CANDIDATES[:-1], key=lambda c: table[c][s])
        old = table[(0.05, 300)][s]
        print(f"  seed={s:<5}: 新最好 {new_best} = {table[new_best][s]:.5f}"
              f"   旧最好 (0.05,300) = {old:.5f}"
              f"   改善 {old - table[new_best][s]:+.5f}")


if __name__ == "__main__":
    main()
