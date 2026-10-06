"""模型定义：把「当前最好的模型 + 特征配方」集中放在一处（单一事实来源）。

为什么要单独一个文件？
- 阶段 E 之前，模型定义散落在各个实验脚本里 → 一旦要"用最好配置生成提交文件"，
  就得手动把参数**再抄一遍**（抄错一个数字，提交出去的就不是你以为的那个模型）。
- 把「当前最好」收敛成一个函数，`tests/make_submission.py` 直接调用 →
  **实验归实验、提交归提交**，两边不会走偏。

⚠️ 这里的参数都不是拍脑袋写的，每一条都有实验记录支撑（见 `docs/experiments.md`）：

| 参数 | 值 | 依据 |
|---|---|---|
| `depth` | 6 | 实验 #18：扫 2~8，CV 呈 U 形，最低点就是默认值 6 |
| `learning_rate` | 0.025 | 实验 #19：3×3 网格最低点（"学习率越小越好"，但收益递减） |
| `iterations` | 1200 | 实验 #19：与学习率**耦合**，300 轮时明显欠训练 |
| 特征配方 | `QualArea` + `TotalSF` | 实验 #6（交互项对线性模型有效）、#10（`TotalSF` 对树模型翻案） |
| 目标编码列 | `Neighborhood` | 实验 #27（筛选 −0.00264）、#28（5×10 确认 **−0.00226**，标准差同时降低） |

当前 CV = **0.11767**（5×10 折重复交叉验证，实验 #28）。
"""
from __future__ import annotations

from catboost import CatBoostRegressor

# —— 特征配方：要启用的派生特征名（算式见 `preprocess.FEATURE_BUILDERS`）——
BEST_DERIVED_FEATURES = ("QualArea", "TotalSF")

# —— 目标编码列（OOF + 平滑，见 `preprocess.OofTargetEncoder`）——
# 只放 Neighborhood：实验 #28 显示再加 MSSubClass 反而略差（0.11803 vs 0.11767）
BEST_TARGET_ENCODE = ("Neighborhood",)

# —— 超参数：来自阶段 E 的实验结论 ——
BEST_PARAMS = dict(
    iterations=1200,
    learning_rate=0.025,
    depth=6,
    random_seed=42,
    verbose=0,                     # 不打印训练日志
    allow_writing_files=False,     # 不生成 catboost_info/ 目录
)


def make_best_model() -> CatBoostRegressor:
    """当前 CV 最好的模型。

    ⚠️ 只给「生成提交文件 / 最终复现」用。
       做**实验对比**时请在自己的脚本里显式写参数 ——
       不要把"实验中的候选方案"和"当前最好"混用一个函数，
       否则实验记录和实际用的模型会对不上号。
    """
    return CatBoostRegressor(**BEST_PARAMS)
