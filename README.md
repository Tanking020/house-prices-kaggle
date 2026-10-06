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
| C · 特征工程 | ✅ 完成 | 从"配方表"机制试特征：`TotalSF`（线性组合）、`QualArea`（交互项）、阈值型 0/1 标志筛查；**`QualArea` 有效**（实验 #5~#7） |
| D · 模型升级 | ✅ 完成 | 决策树 → 随机森林 → GBDT → **XGBoost / LightGBM / CatBoost 三库对比**；结论：CatBoost 最好，学习率小+树多更稳（实验 #8~#17） |
| E · 调参 + 表示方式 | ✅ 完成 | 单变量扫 `depth`（U 形，默认值已最优）→ 联合扫 `lr × iterations`（**新最好 CV 0.11993**）→ 换种子复核排名稳健 → 原生 `cat_features` 无增益但慢 17 倍（实验 #18~#22） |
| F · 模型融合 + 最终提交 | ✅ 完成 | 加权平均（2 个池）与 Stacking（嵌套 CV）都 **≈0 或负收益** → 不采用（实验 #23~#25）；改用**补特征**：`Neighborhood` 目标编码（OOF+平滑）→ **新最好 CV 0.11767**，提交 v3 **Public LB 0.12393** |
| G · 复盘 | ✅ 完成 | 涨分贡献拆解、负结果清单、方法论收获见 [`docs/experiments.md`](docs/experiments.md) 末尾「复盘（阶段 G）」 |

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
│   ├── glossary.zh.md     # 数据字典中英对照速查
│   └── python_syntax_notes.zh.md  # ⭐ Python 与常用库「读代码」速查（语法/参数怎么看）
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

# 每次提交（版本号可自定，默认 v2）
.\\.venv\\Scripts\\python.exe tests\\make_submission.py v2
.\\.venv\\Scripts\\kaggle.exe competitions submit -c house-prices-advanced-regression-techniques `
  -f submissions\\submission_v2.csv -m "改动说明"
```

> ⚠️ 提交前自查：已 `expm1` 还原、列名 `Id,SalePrice`、1459 行、Id 范围 1461~2919。
> 进度条会往 stderr 输出，PowerShell 里可能显示一片红字，**属正常现象**。

## 学习笔记

完整笔记见 [`docs/study_notes.md`](docs/study_notes.md)，包含：

- **Step 1~10**：从环境配置到提交复盘的完整流程
- **模型章**：每个模型"为什么被发明出来"（演化史视角）
- **难点拓展**：偏差-方差分解、多重共线性 VIF、目标编码平滑、Box-Cox/Yeo-Johnson、梯度提升推导、Shapley 值等 16 个专题；
  其中 **拓展 13.3「随机 K 折自己的坑」** 汇总了本项目踩过的 7 个交叉验证陷阱（含自检清单）
- **附录**：数据泄漏专题、术语速查、常见坑总表、面试高频问题

数据字典中英对照见 [`docs/glossary.zh.md`](docs/glossary.zh.md)。

**代码/语法/库看不懂**（"这函数要传几个参数？参数是什么意思？"）→ 看
[`docs/python_syntax_notes.zh.md`](docs/python_syntax_notes.zh.md)：
教你**怎么看函数参数**、项目用到的 Python/numpy/pandas/sklearn 语法速查、
以及"库要不要背"的答案（不背，但要会查）。

**想知道"项目代码每一句在干什么"** → 看
[`docs/code_walkthrough.zh.md`](docs/code_walkthrough.zh.md)：
**逐行讲解全部 Python 代码**（`src/**` 4 个文件 + `tests/**` 15 个脚本），
含「目录」「覆盖检查表」和 **§〇 语法总索引**（每条语法只详解一次）。

---

## 实验结果

> 持续更新中。最新提交 **Public LB = 0.12393**（v3，本地 CV 0.11767）。
> ⚠️ CV 与 LB 的差在 **±0.005~0.006** 量级内属正常抖动（Public LB 只用约一半测试集），
> **看大方向即可**：CV 从 0.14627 → 0.11767，LB 从 0.14206 → 0.12393，方向一致。

| 阶段 | 做法 | CV (RMSE-log) |
|------|------|---------------|
| 基线 | `Pipeline`（4 类填充 + One-Hot/Ordinal）+ `Ridge(alpha=1)` | 0.14627 ± 0.02960（5×10） |
| + 特征工程 | 加 `QualArea = OverallQual × GrLivArea`、`TotalSF = 地下室+1楼+2楼` | 0.14383 ± 0.02699（Ridge） |
| + 模型升级 | 随机森林(200) → GBDT(300, lr=0.05) → CatBoost(300, lr=0.05) | **0.12263 ± 0.01060** |
| + 调参 | CatBoost `lr=0.025, iterations=1200, depth=6`（并换数据划分复核过） | 0.11993 ± 0.01050 |
| + 模型融合 | 加权平均（两个池）与 Stacking（嵌套 CV） | **≈ 0 或负收益 → 不采用**（基模型误差相关 0.94~0.98） |
| + 目标编码 | `Neighborhood` 的 OOF + 平滑目标编码 | **0.11767 ± 0.00887（当前最好）** |

> ⭐ **复盘（涨分贡献拆解）**："换模型族"（线性 → Boosting）贡献了**七成以上**的涨幅；
> 特征工程与调参各只贡献约 0.002~0.003；融合 ≈ 0。完整拆解与**负结果清单**见
> [`docs/experiments.md`](docs/experiments.md) 末尾。

### ⚠️ 一个值得单独记住的坑：K 折交叉验证

> 这些坑**不报错**，只是悄悄把分数变好看。本项目**全部踩过**，每个都有对应的实验编号。

| # | 坑 | 一句话 | 代价 |
|---|---|---|---|
| ① | **考卷不能变** | 在切分**之前**筛选/清洗数据 → 验证集也变了，“涨分”里混了“题目变简单” | 实验 #30：假象 −0.00828 vs 真相 −0.00049（**差 17 倍**） |
| ② | **折内过拟合** | 同一折的数据既用来造特征又用来训练（如目标编码不做 OOF） | 放进 Pipeline 只能挡**跨折**泄漏，**挡不住这个** |
| ③ | **选择偏差** | 用同一把折挑“最优配置” → 分数偏乐观 | 必须**换一套划分复核排名**（#20） |
| ④ | 单次 K 折均值**本身有噪声** | 差异 < 噪声线（≈0.0006）就不值一提 | 先量化噪声线再谈提升 |
| ⑤ | **聚合口径混用** | “每折平均”≠“OOF 汇总”（0.12347 vs 0.12476） | 报告里必须注明口径 |
| ⑥ | **参数结论有前提** | “depth=6 最优”仅在 iterations=300 时成立 | 记录结论时连前提一起写 |
| ⑦ | CV 与 LB 的差**不是常数** | 差值符号会变（−0.004 → +0.005） | 只比方向与排名，别比绝对值 |

> ⭐ **一句话**：**“分数变好”必须有可解释的来源**（新信息 / 更强的模型）；
> 如果来源是“考卷变简单了”或“我挑了最小值”，那就是假象。
> 📖 完整展开（含代码、判据、自检清单）见 [`docs/study_notes.md`](docs/study_notes.md) **拓展 13.3**。

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
