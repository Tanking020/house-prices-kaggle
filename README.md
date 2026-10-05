# House Prices · 机器学习全流程学习项目

> 这是一个**用于系统学习机器学习完整流程**的 Kaggle 回归练习项目。
>
> **项目定位**：重点不在榜单排名，而在于**把每一步的原理搞清楚、把踩过的坑记录下来**。
> 所以我选择"慢一点、但学透"，并把完整学习笔记一并放在仓库里（[`docs/study_notes.md`](docs/study_notes.md)）。

---

## 项目简介

| 项目 | 内容 |
|------|------|
| **任务** | 根据 79 个房屋特征预测成交价 |
| **数据** | Kaggle House Prices（Ames, Iowa）；训练集 1460 条 / 测试集 1459 条 |
| **特征** | 79 个（36 个数值 + 43 个类别） |
| **评估指标** | RMSE on `log(SalePrice)`（官方口径） |
| **项目性质** | 个人学习项目（数学背景转 AI 的入门练习） |

---

## 项目进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 0 · 数据分析（EDA） | ✅ 完成 | 缺失值语义、偏态、类别基数、特征与目标关系、train/test 一致性；结论见 [`notebooks/01_data_overview.ipynb`](notebooks/01_data_overview.ipynb) |
| A · 造"尺子" | ✅ 完成 | 统一的 5 折 CV + log 空间 RMSE，见 [`src/house_prices/evaluate.py`](src/house_prices/evaluate.py) |
| B · 预处理 + 基线 + 首次提交 | ✅ 完成 | 完整预处理装进 `Pipeline` + `Ridge` 基线，并**首次提交**（Public LB 0.14206）；记录见 [`docs/experiments.md`](docs/experiments.md) |
| C~G · 特征工程 → 模型升级 → 调参 → 融合 → 复盘 | ⬜ 未开始 | 路线图见 [`docs/roadmap.md`](docs/roadmap.md) |

> 📍 完整推进计划见 [`docs/roadmap.md`](docs/roadmap.md)（7 阶段 + 2 里程碑）。

---

## 我学到了什么

> ⚠️ **诚实说明**：下面这些是我**通过系统学习 + 与 AI 讨论提前掌握**的关键点，
> **并不代表全部已亲手实践**。已在本项目中真实验证过的标记 ✅，其余待做到时再标记。
> 原则：**宁可标注清楚状态，也不把"学过"说成"做过"。**

### 一、常见的坑（提前学到的，实践进度见"状态"列）

| 坑 | 后果 | 正确做法 | 状态 |
|------|------|---------|------|
| 把 `NA` 全当成缺失值 | 抹掉了"没有该设施"这一信息 | 查数据字典：列出 `NA` 取值的字段，`NA` 表示"无" | ✅ 已在预处理中实践 |
| 在全量数据上算标准化参数 | **数据泄漏**，CV 分数虚高 | 所有 `fit` 只在训练集，用 `Pipeline` 封装 | ⏳ 待组装 Pipeline |
| Target Encoding 用了全量目标值 | **数据泄漏** | 用 K 折 OOF + 平滑 | ⏳ 未到该阶段 |
| Stacking 不用 OOF 生成元特征 | **严重泄漏**，元模型学到错误权重 | 必须用 K 折 OOF | ⏳ 未到该阶段 |
| 给树模型做标准化 | 白费功夫 | 树只看"大于/小于"，不受量纲影响 | ⏳ 未到该阶段 |
| 用 Label Encoding 编码名义类别 | 给无序类别强加了假顺序 | 名义用 One-Hot，有序用 Ordinal | ✅ 已在编码中实践 |
| 填充时只看均值 | 右偏数据被带偏 | 偏态数据用**中位数** | ✅ 已在预处理中实践 |
| 看到异常值就删 | 丢掉真实信息 | 先记录，用 CV 验证后再决定 | ✅ 已在 EDA 中实践（只标记未删） |
| 提交时忘了 `expm1` | 提交的是对数值，分数爆炸 | 训练在 log 空间，提交必须变回原尺度 | ✅ 已实践（首次提交，Public LB 0.14206） |

### 二、我认为最重要的几条纪律

1. **先立"尺子"再动手** —— 评估方案（交叉验证 + 指标）必须先定，否则所有"我改进了"都是错觉
2. **一切 `fit` 只在训练集** —— 这是防数据泄漏的铁律
3. **先跑通闭环，再优化各环节** —— 先做出一个能提交的基线，再回头深挖
4. **特征工程决定分数上限** —— 模型只是"在既定特征上榨分"
5. **用 CV 的相对比较做决策** —— 别迷信绝对分数（调参会让它变得乐观）
6. **每一次实验都记录** —— 改了什么 → 分数变化 → 结论，否则学不到东西

### 三、我觉得比较难的模型 / 方法（尚未实践，属学习笔记）

| 模型 / 方法 | 难点在哪 |
|------|---------|
| **SVR** | 用"误差管道 + ε-不敏感损失"代替逐点拟合；靠核技巧处理非线性；对缩放极度敏感 |
| **XGBoost** | 二阶泰勒展开 + 显式正则化；最优叶子权重与分裂增益都有闭式解 |
| **LightGBM** | 直方图分桶 + Leaf-wise 生长，快但小数据上容易过拟合 |
| **CatBoost** | "有序目标统计"如何做到抗泄漏地处理类别特征 |
| **Boosting 的本质** | 梯度提升 = 在**函数空间**做梯度下降；平方损失下的负梯度恰好就是残差 |
| **贝叶斯优化** | 代理模型（高斯过程同时给出预测值和不确定性）+ 采集函数平衡"利用 vs 探索" |
| **Stacking / OOF** | 为什么必须用 OOF，否则元模型会被"背过答案"的基模型骗到 |

---

## 项目结构

```
house-prices-kaggle/
├── configs/               # 配置文件
├── data/
│   ├── raw/               # 原始数据（未纳入版本控制）
│   └── processed/         # 处理后数据
├── docs/
│   ├── roadmap.md         # ⭐ 完整路线图（7 阶段 + 2 里程碑）
│   ├── study_notes.md     # ⭐ 完整学习笔记（10 步流程 + 难点拓展）
│   ├── experiments.md     # 实验记录（改了什么 → CV 分数 → 结论）
│   └── glossary.zh.md     # 数据字典中英对照速查
├── notebooks/             # 探索性分析（01 · 数据总览与 EDA）
├── reports/figures/       # 图表输出
├── src/house_prices/      # 可复用代码（data / evaluate / preprocess）
├── submissions/           # 提交文件
└── tests/                 # 可重跑的验证脚本
```

## 如何运行

### 1. 准备环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 准备数据

数据来自 Kaggle 竞赛 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)。
出于竞赛规则与仓库体积考虑，**数据未纳入版本控制**，请自行下载后放到：

```
data/raw/
├── train.csv
├── test.csv
├── sample_submission.csv
└── data_description.txt
```

### 3. 提交成绩（Kaggle CLI）

Kaggle 网页的提交页偶尔加载不出上传框（只显示一行 `Need help making a submission? ...` 占位文字）。
改用官方 CLI 更稳、也留得下痕迹：

```powershell
# 首次：浏览器登录（会打开网页授权，只需做一次）
.\.venv\Scripts\kaggle.exe auth login

# 每次提交
.\.venv\Scripts\kaggle.exe competitions submit house-prices-advanced-regression-techniques `
  -f submissions\submission_v1.csv -m "改动说明"
```

> ⚠️ 提交前自查：已 `expm1` 还原、列名 `Id,SalePrice`、1459 行、Id 范围 1461~2919。
> 进度条会往 stderr 输出，PowerShell 里可能显示一片红字，**属正常现象**。

## 学习笔记

完整笔记见 [`docs/study_notes.md`](docs/study_notes.md)，包含：

- **Step 1~10**：从环境配置到提交复盘的完整流程
- **模型章**：每个模型"为什么被发明出来"（演化史视角）
- **难点拓展**：偏差-方差分解、多重共线性 VIF、目标编码平滑、Box-Cox/Yeo-Johnson、梯度提升推导、Shapley 值等 15 个专题
- **附录**：数据泄漏专题、术语速查、常见坑总表、面试高频问题

数据字典中英对照见 [`docs/glossary.zh.md`](docs/glossary.zh.md)。

---

## 实验结果

> 🚧 持续更新中。首次提交 **Public LB = 0.14206**（本地 CV 0.14627，二者接近 → 无泄漏迹象）。

| 阶段 | 做法 | CV (RMSE-log) |
|------|------|---------------|
| 基线 | `Pipeline`（4 类填充 + One-Hot/Ordinal）+ `Ridge(alpha=1)` | 0.14627 ± 0.02960（5×10） |
| + 特征工程 | | |
| + 模型升级 | | |
| + 模型融合 | | |

## 待办

- [x] 数据分析（EDA）
- [x] 统一的评估方案（5 折 CV 尺子）
- [x] 缺失值处理与类别编码
- [x] 组装 `Pipeline`，建立基线模型
- [x] 完成首次 Kaggle 提交（打通闭环 · 里程碑 1）
- [ ] 特征工程与离群点验证
- [ ] 升级模型（XGBoost / LightGBM / CatBoost）并调参
- [ ] 模型融合（OOF Stacking）
- [ ] 补充实验结果与复盘（里程碑 2）
