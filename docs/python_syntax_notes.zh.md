# House Prices 项目 · Python 与常用库「读代码」速查

> **这份笔记解决什么问题**：你已经懂了项目的**步骤**（EDA → 造尺子 → 预处理 → 基线 → …），
> 但打开代码时被**语法**和**库的调用方式**挡住——"这个函数要传几个参数？参数是什么意思？我该传什么？"
>
> **怎么用**：
> 1. 第 0、1 节**必读**（它教你一套"看任何函数都不慌"的方法，比背语法有用）。
> 2. 第 2~6 节是**速查**，不是让你一次读完的课文——**读到代码卡住时，回来对号入座**。
> 3. 第 8 节是可以打印出来贴墙上的**一页速查卡**。
>
> **配套**：ML 概念（为什么这么做）看 [study_notes.md](./study_notes.md)；推进计划看 [roadmap.md](./roadmap.md)。

---

## 0. 先回答你最关心的问题：库要背下来吗？

**不用背。而且"背参数"是学 ML 最没用的一种努力。**

理由：`sklearn` 一个 `RandomForestRegressor` 就有 17 个参数，`LightGBM` 有 100 多个。
没有人靠背记住它们，真正的工程师也是**用到再查**。

那要记住什么？——**只记两样东西**：

| 要记 | 不要记 |
|------|--------|
| ① **"哪类问题 → 找哪个库/哪个对象"**（索引能力） | 具体参数名和默认值 |
| ② **通用动词的语义**（`fit` / `transform` / `predict` 是干嘛的） | 每个类有哪些参数 |

🧒 打个比方：你不需要背下整本字典，只需要知道"要查字去字典、要查路用地图"。
**认路 > 背路。**

第 7 节会给你一张"索引地图"和一套"随手查参数"的流程。

---

## 1. 第一件事：怎么看懂一个函数要传什么

这是全文最重要的一节。学会它，80% 的"看不懂"会自动消失。

### 1.1 函数签名 = 说明书的第一行

Python 里定义一个函数长这样，括号里的部分叫 **签名（signature）**：

```python
def cv_rmse_log(
    model,               # ← 参数：没有 "=" → 必传
    X,
    y,
    n_splits: int = 5,   # ← 参数名 : 类型 = 默认值 → 可省略
    n_repeats: int = 1,
    seed: int = SEED,
    verbose: bool = False,
) -> tuple[float, float]:   # ← "->" 是返回值类型
```

> 这段来自 [evaluate.py](../src/house_prices/evaluate.py) 的 `cv_rmse_log`。

**逐块翻译**：

| 写法 | 名字 | 含义 |
|------|------|------|
| `model` | 参数（无默认值） | **必须传**，不传直接报错 |
| `n_splits: int = 5` | 带默认值的参数 | `: int` 是**类型注解**；`= 5` 表示**不传就用 5** |
| `-> tuple[float, float]` | 返回值注解 | 这个函数**返回两个 float**（一个元组） |

⚠️ **类型注解只是"注释"，Python 运行时不会强制检查**。
写 `n_splits: int` 是给人和工具（Pylance）看的，你传个字符串它也照跑（然后可能崩）。
它的价值：**让你一眼知道"这个位置该放数字还是字符串还是对象"。**

### 1.2 位置参数 vs 关键字参数（怎么传）

```python
# 写法 A：按位置传（顺序必须对得上）
mean, std = cv_rmse_log(ridge, X, y)

# 写法 B：关键字传（写 "参数名=值"，顺序随便）
mean, std = cv_rmse_log(model=ridge, X=X, y=y, n_repeats=10)
```

**怎么选？**（这也是我写代码时遵守的规矩）

| 场景 | 建议 |
|------|------|
| 参数少、含义明显（就像 `cv_rmse_log(model, X, y)`） | 位置传，简洁 |
| 参数多、或有"可选开关" | **后面的用关键字传**，一眼看出改了什么 |

```python
# ✅ 好例子（来自 run_baseline.py）：读的人立刻知道"我把重复次数调成 10 了"
cv_rmse_log(ridge, X, y, n_repeats=10)
```

⚠️ **坑**：一旦你用了一个关键字参数，**它后面的参数也必须用关键字**（顺序规则）。
所以本项目里凡是"只是加一个开关"的场景，都写成 `xxx(..., n_repeats=10)`。

### 1.3 三种"查参数含义"的姿势（按推荐顺序）

**姿势 1：让编辑器告诉你（最快，日常就用这个）**

- **鼠标悬停**在函数名上 → 弹出签名 + 说明。
- 在括号里按 **`Shift + Tab`**（Jupyter）或 **`Ctrl + 空格`**（VS Code）→ 弹出参数提示。
- **`Ctrl + 点击`函数名** → 跳到定义处，直接读源码和注释。

> 本项目所有函数都写了**中文 docstring**（三引号那一段），就是给你当说明书用的。
> 例如打开 [evaluate.py](../src/house_prices/evaluate.py)，`cv_rmse_log` 上方那段
> `参数 / 用法 / ⚠️ 注意` 就是现成的答案。

**姿势 2：`help()`——不离开代码就能打印说明书**

```python
help(pd.read_csv)          # 打印 pandas 的官方 docstring
help(Ridge)                # 打印 Ridge 的参数和解释
```

在 Jupyter 里还能用更短的写法：

```python
pd.read_csv?     # 显示文档
pd.read_csv??    # 连源码一起显示
```

**姿势 3：官方文档（最权威，查"为什么"和完整参数表）**

- scikit-learn：<https://scikit-learn.org/stable/modules/classes.html>
- pandas：<https://pandas.pydata.org/docs/reference/index.html>
- numpy：<https://numpy.org/doc/stable/reference/index.html>

🧒 **读英文文档的小技巧**：不要从头读，只看两处——
① 顶部**一行摘要**（这函数是干嘛的）；
② **Parameters 里你真正要传的那几个**。其余参数先无视。

### 1.4 一套通用套路：拿到任何陌生函数，按这 5 步看

```text
1. 它属于哪个库/对象？  →  pandas / sklearn / 本项目
2. 名字像什么动作？      →  read_ / build_ / fit / to_ ...
3. 看签名：哪些必传、哪些有默认值（= 号）
4. 看每个参数的类型注解 → 该传 DataFrame？字符串？数字？
5. 看返回值 -> 是什么   →  它是"改数据"还是"返回新值"？
```

**第 5 步特别重要**，因为 Python 里一个常见困惑是："这个函数到底改了我原来的变量没有？"

| 行为 | 例子 | 记住 |
|------|------|------|
| **返回新对象**（原数据不动） | `train.drop(columns=[...])`、`np.log1p(y)` | 要用 `=` 接住返回值 |
| **原地修改**（Rare，一般靠 `inplace=True`） | 极少用；本项目一律用"返回新值"的写法 | 见到 `inplace` 才警觉 |

```python
# ✅ 项目实际写法（make_submission.py）：接着返回值
X = train.drop(columns=["Id", "SalePrice"])

# ❌ 如果写成下面这样，X 还是原样，白写！
train.drop(columns=["Id", "SalePrice"])
```

⚠️ 这就是"看不懂为什么没生效"的头号原因：**忘了用变量接住返回值**。

---

## 2. Python 语法骨架（本项目真正用到的那些）

> 只讲**这个项目里出现过**的语法。够你读完整个仓库。

### 2.1 import：两种写法

```python
import pandas as pd                      # 用 pd 当"小名"（alias）
from pathlib import Path                 # 只从 pathlib 里拿 Path 这一个名字
from sklearn.linear_model import Ridge   # 从很深的路径里拿 Ridge
from house_prices.evaluate import cv_rmse_log   # 拿本项目自己写的函数
```

- `import xxx as yy`：给模块起**小名**，后面写 `pd.read_csv` 而不是 `pandas.read_csv`。
- `from a.b import c`：只拿 `c`，后面直接写 `c`。**约定**：`pandas`→`pd`、`numpy`→`np`、`matplotlib.pyplot`→`plt`、`seaborn`→`sns`，全世界都这么写。
- `from __future__ import annotations`：让类型注解"延迟计算"，配合新语法（如 `list[float]`）在旧解释器上也安全。**每个文件开头都有一句，不用管它，照抄即可。**

### 2.2 变量与 f-string（插值）

```python
ROOT = Path.cwd()          # 变量就是给一个值起名字
n = 1460

print(f"训练集有 {n} 行，形状是 {train.shape}")   # f"..." 会把 {} 里的值替换进去
print(f"{m0:.5f} ± {s0:.5f}")                    # :.5f = 保留 5 位小数
```

**f-string 格式速记**（本项目高频）：

| 写法 | 效果 |
|------|------|
| `{x:.5f}` | 保留 5 位小数（分数几乎都用它） |
| `{x:,.0f}` | 千分位、不要小数（打印价格用） |
| `{x:3d}` | 占 3 格宽的整数（对齐输出用，见 `split {i:3d}`） |
| `{x:12s}` | 占 12 格宽的字符串（列名对齐） |

### 2.3 函数：定义、默认参数、返回多个值

```python
def rmse(y_true, y_pred) -> float:      # -> float：只返回一个数
    """普通 RMSE（均方根误差）。越小越好。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
```

```python
def get_xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    ...
    return X, y          # 返回两个值 → 其实是返回一个"元组 (X, y)"
```

**接住多个返回值**：

```python
X, y = get_xy(train)              # 一键拆开
train, test = load_raw()          # 同上
mean, std = cv_rmse_log(...)
```

🧒 **元组（tuple）**：把几个值捆成一组，像 `(X, y)`。左边写几个名字就拆几个，**个数必须对上**。

**默认参数**：`def f(x, n=5)` → 调用时 `f(a)` 用 5，`f(a, n=10)` 用 10。本项目大量使用（如 `cv_rmse_log` 的 `n_repeats=1`）。

### 2.4 容器四件套：list / dict / set / tuple

```python
# list 列表：有序、可重复  —— 用 []
cols = ["OverallQual", "GrLivArea"]
cols.append("GarageCars")            # 末尾加一个

# dict 字典：键 → 值   —— 用 {}
NA_MEANS_NONE = {"PoolQC", "Alley"}  # （这是 set，见下）
checks = {"行数 == 1459": True, "无缺失值": False}   # 真·dict：名字→布尔

# set 集合：无序、去重   —— 用 {}（但里面没有 k: v）
ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]
set(ORDER) & {"TA", "Ex"}            # 交集，用来找"公共的元素"

# tuple 元组：有序、不可改 —— 用 ()
("median", "count")
```

**本项目最常用的集合运算**（来自 notebook 的 train/test 一致性问题）：

```python
only_train = set(train.columns) - set(test.columns)   # 差集：只在 train 里的列
overlap = set(train["Id"]) & set(test["Id"])          # 交集：两边都有的 Id
```

| 运算 | 符号 | 意思 |
|------|------|------|
| 交集 | `a & b` | 两边都有 |
| 并集 | `a \| b` | 合起来去重 |
| 差集 | `a - b` | 在 a、不在 b |

### 2.5 推导式（一行写出一个 list / dict）

```python
# 语法：[ 表达式  for 元素 in 可迭代对象  if 条件 ]
num_cols = [c for c in X.columns if c not in excludes]
ordinal   = [c for c in ORDINAL_COLS if c in X.columns]
```

🧒 读法："对 `X.columns` 里的每个 `c`，如果 `c` 不在 `excludes` 里，就把 `c` 收进新列表。"

字典推导（本项目也偶尔用）：

```python
pred_map = {i: p for i, p in zip(ids, prices)}   # 形式：[key: value for ...]
```

⚠️ 推导式看着"高级"，其实就是**一个 for 循环 + append 压成一行**。看不懂时，先在脑子里展开成普通循环。

### 2.6 if / for / enumerate / 循环解包

```python
for i, (tr_idx, va_idx) in enumerate(splitter.split(X), start=1):
    ...
```

拆开读：
- `splitter.split(X)` 每次产出 `(tr_idx, va_idx)` 一对；
- `(tr_idx, va_idx)` 把这个**元组拆成两个变量**；
- `enumerate(..., start=1)` 额外给出**编号 `i`**，从 1 开始（默认从 0）。

```python
if n_repeats == 1:          # 相等比较用 ==，不是 =（= 是赋值！）
    splitter = KFold(...)
else:
    splitter = RepeatedKFold(...)
```

常用的比较/逻辑：

| 写法 | 含义 |
|------|------|
| `==` / `!=` | 等于 / 不等于 |
| `>` `<` `>=` `<=` | 大小 |
| `and` / `or` / `not` | 与 / 或 / 非 |
| `x in a` | x 在 a 里吗 |
| `a is None` | **是**那个空对象吗（比 `== None` 更规范） |

### 2.7 内置函数速记表（本项目用到的）

| 函数 | 作用 | 项目里的例子 |
|------|------|--------------|
| `len(x)` | 长度/个数 | `len(train)`、`len(sample)` |
| `print(x)` | 打印 | 到处 |
| `sorted(iterable)` | 排序并返回新列表 | `sorted(X_imp["MasVnrType"].unique())` |
| `set(iterable)` | 变成集合（去重） | `set(train.columns)` |
| `int(x)` / `float(x)` / `str(x)` | 类型转换 | `int(X.isna().sum().sum())` |
| `range(n)` | 0..n-1 的序列 | `range(10)` |
| `hasattr(obj, "iloc")` | 有没有这个属性 | `_take_rows` 里判断是不是 DataFrame |
| `zip(a, b)` | 两个列表"拉链"配对 | `zip(ids, prices)` |
| `type(x).__name__` | 取类型名字符串 | `type(pipe.named_steps['model']).__name__` |

### 2.8 Path：路径不是字符串

```python
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
train = pd.read_csv(RAW_DIR / "train.csv")
```

逐段翻译（**这是全项目最"唬人"的一行，拆开就没事**）：

| 片段 | 含义 |
|------|------|
| `__file__` | 当前 Python 文件自己的路径（一个字符串） |
| `Path(__file__)` | 把它变成 `Path` 对象 |
| `.resolve()` | 变成**绝对路径**（把 `..` 之类算清楚） |
| `.parents[2]` | 往上跳 2 层目录（`parents[0]`=上一层，`parents[1]`=上两层…） |
| `/ "data" / "raw"` | **Path 的 `/` 是"拼路径"**，不是除法！ |

🧒 为什么不用字符串拼？因为 Windows 用 `\`、Linux 用 `/`，`Path` 会**自动处理**，还顺带避免了拼错。
把 `"data"` 理解成**文件夹名字**就行。

常用的 Path 方法：

| 写法 | 作用 |
|------|------|
| `ROOT / "submissions" / "a.csv"` | 拼路径 |
| `OUT_PATH.parent` | 父目录 |
| `OUT_PATH.parent.mkdir(parents=True, exist_ok=True)` | 建目录（已存在也不报错） |
| `path.exists()` | 存在吗 |
| `path.relative_to(ROOT)` | 相对路径（打印时好看） |

### 2.9 `if __name__ == "__main__":`

```python
def main() -> None:
    ...

if __name__ == "__main__":
    main()
```

🧒 意思是："**只有当这个文件被直接运行时**，才执行 `main()`；被别人 `import` 时不执行。"
这样文件既能当**脚本**跑，又能当**模块**被导入，互不打架。本项目每个 `tests/*.py` 都有这两行。

### 2.10 本项目的"脚本头三件套"

几乎每个 `tests/*.py` 开头都是这三句，**照抄即可**：

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))        # 让脚本能找到 src/house_prices

from house_prices.data import get_xy, load_raw   # noqa: E402
```

**为什么需要**：`src/house_prices` 不是"装好的库"，只是仓库里的一个文件夹。
要在 `tests/` 的脚本里 `import house_prices`，得先把 `src/` **告诉 Python**——
`sys.path.insert(0, ...)` 就是把 `src/` 加到"去哪找模块"的清单最前面。

`# noqa: E402`：告诉 linter"我知道这行 import 没写在文件最顶部，别报警"（因为要先设置好 `sys.path` 才能导入）。

⚠️ 这也是本项目"**库 vs 脚本**"分工的由来（协作规范里的铁律）：
`src/house_prices/*.py` 是**库**（只定义函数，直接运行没输出）；`tests/*.py` 是**脚本**（有 `main()`，能跑给你看结果）。

---

## 3. numpy：把"一整列数字"当成一个整体来算

🧒 numpy 的核心对象叫 `ndarray`——**一个装着同类型数字的盒子**，可以整体加减乘除，不用写循环。

```python
import numpy as np

y = np.asarray(y, dtype=float)         # 把别的容器转成 numpy 数组，并确保是小数
np.log1p(y)                            # log(1+y)，逐元素计算
np.expm1(z)                            # exp(z)-1，逐元素计算（log1p 的逆运算）
np.sqrt(np.mean((y_true - y_pred) ** 2))   # 先逐元素相减、平方，再求平均，最后开根
np.mean(scores) / np.std(scores)       # 均值 / 标准差
```

**为什么用 `log1p` / `expm1` 而不是 `np.log` / `np.exp`？**
因为价格可能接近 0，`log(0)` 会变成 `-inf`；`log1p(x)=log(1+x)` 在 0 附近更稳。**记住它们互为逆运算**：

```python
np.expm1(np.log1p(y)) == y     # 近似成立（浮点误差内）
```

**向量化（vectorization）**：`y_true - y_pred` 这种"整盒减整盒"的写法叫向量化。
⚠️ 它**不等于** Python 的 for 循环，但结果一样、速度快很多。看到数组之间的 `+ - * / **`，就理解成"**逐个元素**做运算"。

**`np.float64` → `float()`**：numpy 的数字类型自带小尾巴，本项目习惯在最后 `float(...)` 转成普通 Python 数字，打印更干净。

---

## 4. pandas：会算数的 Excel 表

🧒 两个核心对象：

| 对象 | 类比 | 本项目例子 |
|------|------|-----------|
| `DataFrame` | 一张**表**（有行有列、列有名字） | `train`、`test`、`X` |
| `Series` | 表里的**一列**（或一行） | `train["SalePrice"]`、`y` |

> ⚠️ 本项目环境是 **pandas 3.0**：字符串列的 dtype 显示为 `str`（老教程里常见的是 `object`）。**意思一样，别被版本差异吓到。**

### 4.1 读进来 / 写出去

```python
train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")   # 读 CSV → DataFrame
sample.to_csv(OUT_PATH, index=False)                       # 写出 CSV
```

⚠️ `index=False` 是**必须**的：否则会多写一列 pandas 的行号，Kaggle 提交直接报废。

### 4.2 看数据长什么样（EDA 高频）

```python
train.shape                # (1460, 81)  行数、列数
train.dtypes               # 每一列的类型
train.dtypes.value_counts()  # 每种类型各有几列
train.head(5) / .tail(8)   # 前 5 行 / 后 8 行
train.columns              # 列名列表（Index 对象）
train["SalePrice"].describe().round(0)   # 计数/均值/标准差/分位数/最值
train["SalePrice"].skew()  # 偏度（>0 = 右偏）
train.nunique()            # 每列有几个不同取值
train["Id"].is_unique      # 这一列是不是每行都不同
```

### 4.3 选列、丢列、按类型筛

```python
y = train["SalePrice"]                     # 选一列 → Series
X = train.drop(columns=["Id", "SalePrice"])  # 丢掉这些列 → 新 DataFrame

num_cols = train.select_dtypes(include="number").columns.tolist()   # 数值列
cat_cols = train.select_dtypes(exclude="number").columns.tolist()   # 非数值(类别)列
X[["OverallQual", "GrLivArea"]]            # 选多列 → 用双括号 [["a","b"]]
```

⚠️ 单括号 `["a"]` 取一列（Series），双括号 `[["a","b"]]` 取多列（DataFrame）。这是新手最常见的困惑之一。

### 4.4 缺失值

```python
X.isna()                  # 每个格子"是不是空"的布尔表（True/False）
X.isna().sum()            # 每列空了几个
X.isna().sum().sum()      # 全表总共空了几个
X.isna().mean() * 100     # 每列的缺失百分比
```

`isna()` → `sum()` → `mean()` 这种"**一个接一个**"的写法叫**链式调用**。

### 4.5 链式调用怎么读（重要）

```python
corr = (
    train[num_cols]                 # 1) 挑出数值列
    .corr()["SalePrice"]            # 2) 算相关系数矩阵，取 SalePrice 那一列
    .drop(["Id", "SalePrice"])      # 3) 去掉 Id 和 SalePrice 自己
    .sort_values(ascending=False)   # 4) 从大到小排
)
```

**读法：从上往下，每一行吃上一行的输出。** 相当于：

```python
a = train[num_cols]
b = a.corr()["SalePrice"]
c = b.drop(["Id", "SalePrice"])
corr = c.sort_values(ascending=False)
```

🧒 括号 `(...)` 包住整条链，只是为了**换行写更好看**，没有别的含义。

### 4.6 排序 / 取子集 / 分组统计

```python
miss = train.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)   # 只留"有缺失的列"，从多到少

train.groupby("OverallQual")["SalePrice"].median()   # 按 OverallQual 分组，求各组房价中位数
train["GrLivArea"] > 4500                            # 一列布尔值（条件）
outliers = train[train["GrLivArea"] > 4500]          # 用布尔值"筛行"

nb = (train.groupby("Neighborhood")["SalePrice"]
      .agg(median="median", count="count")           # 一次算多个统计量，并起名字
      .sort_values("median"))
nb["median"].iloc[0]                                 # 取第 0 个（按位置，不是按名字）
nb.index[0]                                          # 取第 0 个的行标签
```

| 写法 | 含义 |
|------|------|
| `df[df["col"] > 5]` | **布尔筛选**：只保留条件为 True 的行 |
| `df.groupby("k")["v"].median()` | 按 k 分组，对 v 求中位数 |
| `.agg(median="median", count="count")` | 一次算多个统计量，并给结果列起名 |
| `.iloc[0]` | 按**位置**取 |
| `.loc["Gd"]` / `.reindex(ORDER)` | 按**标签**取 / 按指定标签顺序重排 |
| `.value_counts()` | 每个取值出现多少次 |
| `.map(字典或Series)` | 按"键→值"逐行替换（提交时按 Id 对齐就靠它） |
| `.duplicated(subset=feat)` | 这些列组合起来有没有重复行 |

### 4.7 构造新表

```python
overview = pd.DataFrame({                 # 用"字典"建表：{列名: 一列数据}
    "dtype": train.dtypes.astype(str),
    "n_unique": train.nunique(),
    "n_missing": train.isna().sum(),
})
pred_map = pd.Series(prices, index=test["Id"])   # 用 Id 当索引的一列
sample["SalePrice"] = sample["Id"].map(pred_map) # 按 Id 把预测值映射回去
```

🧒 `pd.DataFrame({...})` 里，**字典的 key 变成列名，value 变成那一列**。

---

## 5. matplotlib / seaborn：画图三件套

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(11, 4))     # 一张画布，1 行 2 列两个坐标轴
sns.histplot(train["SalePrice"], kde=True, ax=axes[0])   # 画到第 0 个坐标轴上
axes[0].set_title("原始 SalePrice")
sns.histplot(np.log1p(train["SalePrice"]), ax=axes[1], color="seagreen")
axes[1].set_title("log1p(SalePrice)")
plt.tight_layout()    # 自动调间距，别让字挤在一起
plt.show()            # 显示出来
```

| 写法 | 含义 |
|------|------|
| `plt.subplots(1, 2, figsize=(11,4))` | 建画布，返回 `(fig, axes)`；两个子图时 `axes` 是个含 2 个坐标轴的数组 |
| `ax=axes[0]` | 指定"画到哪个子图上" |
| `ax.set_title("...")` | 给某个子图加标题 |
| `plt.tight_layout()` / `plt.show()` | 排版 / 出图 |

散点图（找离群点用的）：

```python
sns.scatterplot(data=train, x="GrLivArea", y="SalePrice", alpha=0.5, ax=ax)
```

**中文字体**（Windows 上不设置会显示成方块）：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False     # 让负号能正常显示
```

> ⚠️ 这两行是**环境设置**，跑一次即可（本项目写在 notebook 的"准备"单元里）。

---

## 6. scikit-learn：记 4 个动词 + 3 个容器 + 一张参数表

sklearn 是"ML 工具箱"，它的设计**极其统一**——所有模型/变换器都遵守同一套接口。**记住这套接口，比记任何参数都值钱。**

### 6.1 四个动词（最重要的心法）⭐

| 动词 | 中文 | 谁有它 | 干什么 |
|------|------|--------|--------|
| `fit` | 学习 / 拟合 | 所有模型、所有变换器 | **从数据里学参数**（如均值、中位数、系数） |
| `transform` | 转换 | 变换器（Imputer / Encoder…） | **用学到的参数处理数据**，产出新的特征矩阵 |
| `fit_transform` | 上面两步合一 | 变换器 | 等价于 `fit` 完再 `transform`（更省事） |
| `predict` | 预测 | 模型（回归器/分类器） | 给定 X，输出预测值 |

```python
# 变换器：fill → transform  （见 smoke_preprocess.py）
imp = build_imputer(X)
X_imp     = imp.fit_transform(X)        # 在 train 上：学习 + 填充
X_test_imp = imp.transform(X_test)      # 在 test 上：只用 train 学到的规则填充

# 模型：fit → predict  （见 smoke_pipeline.py）
pipe.fit(X, np.log1p(y))     # 训练
pred_log = pipe.predict(X_test)   # 预测
```

⚠️ **铁律（本项目红线）**：`fit` **只能在训练集**上做！
验证集/测试集**只允许 `transform` / `predict`**。
一旦在 test 上 `fit`，就是**数据泄漏**——CV 分数虚高，提交就崩。
🧒 人话：**"考前把答案偷偷背进模型"** 就是泄漏；`Pipeline` 是把"考生+答题规则"打包在一起、保证规则只在训练时学。

### 6.2 三个容器

| 容器 | 一句话 | 项目里的例子 |
|------|--------|--------------|
| **Estimator**（估计器） | 任何有 `fit/predict` 的模型，如 `Ridge()` | `Ridge(alpha=1.0)` |
| **Transformer**（变换器） | 任何有 `fit/transform` 的预处理件 | `SimpleImputer`、`OneHotEncoder` |
| **Pipeline** | 把"多步预处理 + 模型"**串成一条线** | `make_pipeline(Ridge(), X)` |

```python
# Pipeline 的写法：一串 (名字, 对象) 的列表（见 preprocess.py）
Pipeline([
    ("impute", build_imputer(X)),
    ("encode", build_encoder(X)),
    ("model", model),
])
```

- `("impute", 对象)`：`"impute"` 只是**你给它起的小名**，方便以后取用。
- `ColumnTransformer`：**同一张表的不同列走不同处理**时用它：

```python
ColumnTransformer([
    ("cat_none", SimpleImputer(strategy="constant", fill_value="None"), cat_semantic),
    ("num_median", SimpleImputer(strategy="median"), other_num),
], remainder="drop")
```

读法：**`(小名, 用哪个变换器, 处理哪些列)`**。`remainder="drop"` = 没被点名的列直接丢掉。

### 6.3 项目里所有 sklearn 参数，逐条翻译

| 代码 | 参数 | 人话含义 |
|------|------|----------|
| `SimpleImputer(strategy="median")` | `strategy` | 用什么填：`"median"` 中位数 / `"most_frequent"` 众数 / `"constant"` 固定值 |
| `SimpleImputer(strategy="constant", fill_value="None")` | `fill_value` | `strategy="constant"` 时具体填什么 |
| `OneHotEncoder(handle_unknown="ignore")` | `handle_unknown` | 遇到"训练集没见过的类别"时**别报错**，编码成全 0 |
| `OneHotEncoder(sparse_output=False)` | `sparse_output` | 输出**普通稠密数组**（本项目要配合 pandas 输出，所以关掉稀疏） |
| `OrdinalEncoder(categories=[QUALITY_ORDER] * len(ordinal))` | `categories` | 手动指定**等级顺序**，保证 `None<Po<Fa<TA<Gd<Ex` 不乱 |
| `KFold(n_splits=5, shuffle=True, random_state=42)` | `n_splits` / `shuffle` / `random_state` | 切几折 / 切之前打乱 / 固定随机种子（可复现） |
| `RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)` | `n_repeats` | 重复几轮 5 折（更稳） |
| `Ridge(alpha=1.0)` | `alpha` | 正则化强度，越大越"保守"（简单说：越不容易过拟合、但越可能欠拟合） |
| `DummyRegressor(strategy="mean")` | `strategy` | "什么都不学"的模型：只输出训练集均值（当**地板**用） |
| `clone(model)` | — | **复刻一个同样的模型**（每折都用全新模型，防止带上一折的记忆） |
| `df.set_output(transform="pandas")` | `transform` | 让变换器**输出 pandas DataFrame**（保留列名，方便调试） |
| `prep.get_feature_names_out()` | — | 看编码后**每列叫什么名字**（诊断列数膨胀） |

### 6.4 交叉验证的写法长什么样

```python
splitter = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (tr_idx, va_idx) in enumerate(splitter.split(X), start=1):
    m = clone(model)
    m.fit(X.iloc[tr_idx], np.log1p(y[tr_idx]))     # 训练部分
    pred = m.predict(X.iloc[va_idx])               # 验证部分
    ...
```

- `splitter.split(X)` 每次**吐出两个"行号清单"**：`tr_idx`(训练行) 和 `va_idx`(验证行)。
- `X.iloc[tr_idx]`：按行号取出那些行。
- ⚠️ **`split(X)` 只用到 X**——为什么要传 X？因为这题是**普通 K 折**（每行独立），它只需要知道"有多少行"。若同一实体有多行，才需要 `GroupKFold`。**本项目已核验每行是独立房子，用普通 KFold 即可。**

---

## 7. 库不用背，但需要一张"索引地图"

把下面这张表当成**"我要做 X → 用 Y"** 的目录。记不住细节没关系，**记得"有这么个东西"就行**。

| 我想干的事 | 找谁 |
|-----------|------|
| 读/写 CSV | `pd.read_csv` / `df.to_csv(index=False)` |
| 看表有多大、什么类型 | `df.shape` / `df.dtypes` / `df.head()` |
| 挑数值列 / 类别列 | `df.select_dtypes(include=...)` / `exclude=...` |
| 数缺失值 | `df.isna().sum()` |
| 分组求统计 | `df.groupby("k")["v"].median()` / `.agg(...)` |
| 算相关性 | `df.corr()["SalePrice"]` |
| 排序 / 取前几行 | `.sort_values(...)` / `.head(n)` |
| 填缺失值 | `sklearn.impute.SimpleImputer` |
| 类别 → 数字 | `OneHotEncoder`（无序）/ `OrdinalEncoder`（有序） |
| 串起预处理 | `Pipeline` + `ColumnTransformer` |
| 切数据做交叉验证 | `sklearn.model_selection.KFold` / `RepeatedKFold` |
| 线性模型 | `sklearn.linear_model.Ridge` / `LinearRegression` / `Lasso` |
| 树模型（以后用） | `sklearn.ensemble.RandomForestRegressor`、`xgboost`、`lightgbm`、`catboost` |
| 算 RMSE | 自己写（见 `evaluate.py`）或 `sklearn.metrics.mean_squared_error` |
| 画分布 / 散点 | `sns.histplot` / `sns.scatterplot` |

**"随手查参数"的标准流程**（背不下来就用它）：

```text
1. 编辑器里悬停函数名 → 看签名 + 一行摘要
2. 还不够？Shift+Tab（Jupyter）/ Ctrl+空格（VS Code）看参数提示
3. 还不行？help(对象) 或 打开官方文档，只看 "Parameters" 里你要传的那几个
4. 搞不清"输出长啥样"？直接跑一个 3 行的小例子，print 出来看
```

> 💡 **第 4 条是这个项目最提倡的学法**：不确定就**跑一小段验证**，用输出说话，而不是猜。
> （见 `tests/smoke_*.py`——它们就是"跑一小段、看结果"的范例。）

---

## 8. 一页速查卡（可打印）

### Python 骨架

```python
import pandas as pd                  # 小名
from pathlib import Path             # 只拿一个名字
X, y = get_xy(train)                 # 拆元组
[c for c in cols if c in keep]       # 列表推导
f"{x:.5f}  {n:,.0f}"                 # f-string 格式
if __name__ == "__main__": main()    # 脚本入口
ROOT = Path(__file__).resolve().parents[1]   # 仓库根目录
```

### 四个动词

| 动词 | 用在哪 | 只在训练集？ |
|------|--------|--------------|
| `fit` | 学参数 | ✅ 只能 train |
| `transform` | 用参数处理数据 | 验证/测试集 |
| `fit_transform` | 两步合一 | ✅ 只能 train |
| `predict` | 出预测 | 任意（用已 fit 的模型） |

### pandas 高频十连

```python
df.shape / df.dtypes / df.head()
df["col"] / df[["a","b"]] / df.drop(columns=[...])
df.select_dtypes(include="number")
df.isna().sum() / df.nunique() / df.skew()
df["col"].value_counts()
df[df["col"] > 0]
df.groupby("k")["v"].median()
df.sort_values("col", ascending=False).head(10)
df["col"].iloc[0] / df["col"].map(d)
df.corr()["SalePrice"] / df.to_csv(path, index=False)
```

### 项目约定（别忘了）

```python
np.log1p(y)   # 训练前：目标取对数
np.expm1(p)   # 提交前：还原回价格（⚠️ 忘了分数爆炸）
to_csv(path, index=False)     # 提交不写索引
```

---

## 9. 自测点与练习

**自测点（能答上来 = 这一步过了）**

1. 看到一个函数 `def f(a, b=3) -> int`，你能说出：`a` 必须传吗？`b` 不传会怎样？返回什么？
2. `X = train.drop(columns=["Id"])` 和单独写 `train.drop(columns=["Id"])` 有什么差别？
3. `fit` 和 `transform` 的区别是什么？为什么 `test` 只能 `transform`？
4. `Path(__file__).resolve().parents[2]` 里的 `parents[2]` 是往上几层？`/ "data"` 是除法吗？
5. `data/raw` 里的 `train.csv` 有 81 列，`get_xy` 帮你丢掉了哪两列？为什么？

**动手练习（都在 `tests/` 里有现成例子可对照）**

```powershell
# 从项目根目录运行（Windows PowerShell）
.\.venv\Scripts\python.exe tests\smoke_evaluate.py     # 看四个动词 + CV 打分
.\.venv\Scripts\python.exe tests\smoke_preprocess.py   # 看 fit_transform / transform
.\.venv\Scripts\python.exe tests\run_baseline.py       # 看完整流程
```

**练习 A**：打开 [evaluate.py](../src/house_prices/evaluate.py)，**不看我的讲解**，自己说出 `cv_rmse_log` 每个参数的含义；再用 `help()` 或悬停对照检查。
**练习 B**：在 notebook 里跑 `train["SalePrice"].describe()`，然后自己写一行算出"SalePrice 的缺失个数"。
**练习 C**：故意把 `cv_rmse_log(..., n_repeats=10)` 改成 `n_repeats=100`，看分数"稳不稳"——体会 `std` 的含义。

---

> **一句话总结**：**不要背库，要会认动词、会查参数、会跑小例子验证。**
> 卡住时回到第 1 节那 5 个步骤——它适用于本文档之外的**任何**代码。
