"""预处理：把 EDA 的结论变成可复用的 sklearn 变换器。

⚠️ 铁律：这里的变换全都必须放进 Pipeline，
   只在【训练集】上 fit，再 transform 验证集 / 测试集 —— 否则数据泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# —— 列清单来自 EDA（见 notebooks/01_data_overview.ipynb）——

# 语义缺失·分类：NA 表示"没有该设施" → 填 "None"（是信息，不是丢失）
CAT_SEMANTIC_NONE = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# 语义缺失·数值：没有该设施 → 填 0（不能填中位数，那等于捏造信息）
NUM_SEMANTIC_ZERO = ["GarageYrBlt", "MasVnrArea"]


def build_imputer(X) -> ColumnTransformer:
    """构造"只做缺失值填充"的 ColumnTransformer。

    只读取 X 的【列名与类型】来决定分组（不读数值），因此不构成泄漏。

    分成 4 组：
      1) 语义缺失·分类 → 填 "None"
      2) 语义缺失·数值 → 填 0
      3) 其余数值      → 中位数
      4) 其余分类      → 众数
    第 3、4 组是"兜底"：test 里有 15 列缺失而 train 不缺，必须也能被填上。
    """
    cat_semantic = [c for c in CAT_SEMANTIC_NONE if c in X.columns]
    num_semantic = [c for c in NUM_SEMANTIC_ZERO if c in X.columns]

    num_all = X.select_dtypes(include="number").columns.tolist()
    cat_all = X.select_dtypes(exclude="number").columns.tolist()
    other_num = [c for c in num_all if c not in num_semantic]
    other_cat = [c for c in cat_all if c not in cat_semantic]

    ct = ColumnTransformer(
        [
            ("cat_none", SimpleImputer(strategy="constant", fill_value="None"), cat_semantic),
            ("num_zero", SimpleImputer(strategy="constant", fill_value=0), num_semantic),
            ("num_median", SimpleImputer(strategy="median"), other_num),
            ("cat_mode", SimpleImputer(strategy="most_frequent"), other_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")  # 输出保留 DataFrame，方便查看 / 调试
    return ct


# —— 有序评级列（等级有意义：None < Po < Fa < TA < Gd < Ex）——
ORDINAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
# "None" 表示"没有该设施"（由缺失值填充而来），放在最低位
QUALITY_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]


def build_encoder(X) -> ColumnTransformer:
    """构造"只做类别编码"的 ColumnTransformer（输入必须是【已填好缺失】的数据）。

    三路：
      1) 有序评级列 → OrdinalEncoder，按 QUALITY_ORDER 映射成 0~5
      2) 其余类别列 → OneHotEncoder（每个取值一列，不假设顺序）
      3) 数值列     → 原样通过

    `handle_unknown="ignore"`：遇到训练集没见过的类别时编码成全 0，而不是报错。
    """
    ordinal = [c for c in ORDINAL_COLS if c in X.columns]
    cat_all = X.select_dtypes(exclude="number").columns.tolist()
    nominal = [c for c in cat_all if c not in ordinal]
    num_all = X.select_dtypes(include="number").columns.tolist()

    ct = ColumnTransformer(
        [
            # 每一列都用同一套等级顺序（多出来的等级允许存在，只是用不到）
            ("ordinal", OrdinalEncoder(categories=[QUALITY_ORDER] * len(ordinal)), ordinal),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal,
            ),
            ("num", "passthrough", num_all),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


# —— 派生特征"配方表"：名字 → 怎么算 ——
# 用字典把"特征名"和"算式"解耦：想试新特征，只在这里加一行；
# 想开关某个特征，只要在特征名列表里增删即可 → 天然的 A/B 实验机制。
FEATURE_BUILDERS = {
    # 线性组合（总和）：对线性模型零增益，但对树模型有用
    "TotalSF": lambda X: X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"],
    # 交互项（乘积）：线性模型**无法**自己表示 → 能提供新信息
    "QualArea": lambda X: X["OverallQual"] * X["GrLivArea"],
    # —— 以下为"阈值型"特征：把连续值压成 0/1 ——
    # 阈值化也是一种**非线性**变换（模型无法用线性组合造出">0"这种跳变）
    "HasPool": lambda X: (X["PoolArea"] > 0).astype(int),
    "Has2ndFlr": lambda X: (X["2ndFlrSF"] > 0).astype(int),
    "HasBsmt": lambda X: (X["TotalBsmtSF"] > 0).astype(int),
    "HasGarage": lambda X: (X["GarageArea"] > 0).astype(int),
    "HasFireplace": lambda X: (X["Fireplaces"] > 0).astype(int),
    # —— 批次 2（阶段 F 回头补特征）：房龄系 ——
    # 原始数据里只有"年份"（YearBuilt / YrSold），而房价更关心"到卖的时候多少年了"。
    # ⭐ 注意：这类特征是 `YrSold - YearBuilt`，**不是现有列的线性组合**
    #    （现有列里没有 YrSold 与 YearBuilt 的差），所以对线性模型也有效。
    "HouseAge": lambda X: X["YrSold"] - X["YearBuilt"],
    "RemodAge": lambda X: X["YrSold"] - X["YearRemodAdd"],
    # 翻修过没有：YearRemodAdd == YearBuilt 表示"从未翻修"
    "IsRemodeled": lambda X: (X["YearRemodAdd"] != X["YearBuilt"]).astype(int),
    # —— 批次 2：卫浴总数（线性组合 → 对树模型有效，和 TotalSF 同类）——
    "TotalBath": lambda X: (
        X["FullBath"] + 0.5 * X["HalfBath"]
        + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
    ),
}


class AddDerivedFeatures(BaseEstimator, TransformerMixin):
    """按名字添加派生特征（纯函数变换：fit 不学习任何东西，无泄漏）。

    参数 features：要添加哪些特征（名字取自 FEATURE_BUILDERS）。
    传空 = 不加任何特征（A/B 实验的"基线"）。

    ⚠️ 实验纪律：一次只开一个特征，用 CV 验证后再决定留不留。
    """

    def __init__(self, features=()):
        self.features = features

    def fit(self, X, y=None):
        # 只登记列信息（feature_names_in_ / n_features_in_），不学习任何东西。
        # ⚠️ 这一步必须做：sklearn 的 Pipeline 要靠它拼"特征名链条"。
        # 注意：不能用 validate_data —— 它默认要求数值，而 X 里有字符串列；
        # 这里只需手工登记两个属性即可。
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = X.copy()                      # ⚠️ 绝不原地修改传进来的数据
        for name in self.features:
            X[name] = FEATURE_BUILDERS[name](X)
        return X

    def get_feature_names_out(self, input_features=None):
        # 注意：Pipeline 第一步会传 input_features=None，
        # 此时要回退到自己 fit 时登记的 feature_names_in_。
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array(list(input_features) + list(self.features))


class OofTargetEncoder(BaseEstimator, TransformerMixin):
    """目标编码：把类别列换成"该类别的平均 log 价格"（**OOF + 平滑**）。

    ⚠️ 必须解决的两个问题（这也是目标编码最容易翻车的地方）：

    ① **跨折泄漏**：编码统计量里绝不能出现验证集的 y。
       → 本类做成 Pipeline 的一步，`fit` 只会见到**训练折**，天然安全。

    ② **折内过拟合**：如果"用训练折全部行算均值，再拿它训练同一批行"，
       那么每一行的编码里**包含了它自己的 y** → 模型会过度信任这个特征
       （训练集上虚好、验证集上崩）。
       → `fit_transform` 内部**再做一层 K 折**：第 i 折的编码只用【其余折】算出来。

    **平滑**：类别样本少时均值极不稳定（`Neighborhood=Blueste` 只有 2 套房）。
       encoded = (n * mean_c + m * prior) / (n + m)
       n=0 → 完全用 prior（全局均值）；n 很大 → 接近该类别自己的均值。
       m = `smoothing`（"先验相当于多少条虚拟样本"）。

    `transform`（新数据/验证集/测试集）用的则是**全量训练数据**算出的平滑均值。

    📌 sklearn ≥1.3 有内置的 `sklearn.preprocessing.TargetEncoder`（同样带交叉拟合）。
       这里自己写一遍是为了**看清机制**；工程上可以直接用内置的。
    """

    def __init__(self, columns=(), n_splits=5, smoothing=10.0, seed=42):
        self.columns = columns
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed

    # ---- 内部：算"平滑后的类别均值" ----
    def _smoothed_means(self, col, y) -> dict:
        values = np.asarray(col)
        out: dict = {}
        for cat in np.unique(values):
            mask = values == cat
            n = int(mask.sum())
            mean = float(y[mask].mean())
            out[cat] = (n * mean + self.smoothing * self.prior_) / (n + self.smoothing)
        return out

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        # 手工登记 sklearn 约定属性（不能用 validate_data，它会要求全数值）
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.prior_ = float(np.mean(y))
        self.mapping_ = {c: self._smoothed_means(X[c], y) for c in self.columns}
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """训练数据走这里：内部 K 折 → 每行的编码只用别的行算。"""
        if y is None:
            raise ValueError("OofTargetEncoder 需要 y（目标编码必须用到目标）")
        self.fit(X, y)
        y_arr = np.asarray(y, dtype=float)
        X_out = X.copy()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for c in self.columns:
            oof = pd.Series(np.nan, index=X.index, dtype=float)
            for tr_idx, va_idx in kf.split(X):
                means = self._smoothed_means(X[c].iloc[tr_idx], y_arr[tr_idx])
                oof.iloc[va_idx] = X[c].iloc[va_idx].map(means).to_numpy()
            # 该折里没见过的类别 → 回退全局均值
            X_out[c] = oof.fillna(self.prior_).astype(float)
        return X_out

    def transform(self, X):
        X_out = X.copy()
        for c in self.columns:
            mapped = X[c].map(self.mapping_[c]).astype(float)
            X_out[c] = mapped.fillna(self.prior_)      # 训练集没见过的类别 → 全局均值
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def cat_feature_names(X) -> list[str]:
    """列出所有【类别型】列名（= 非数值列）。

    用途：有些库（如 CatBoost）不需要先做 One-Hot，而是**直接吃原始类别列**，
    此时要把这 43（或加上 MSSubClass 就 44）列的列名告诉它。
    只读列名/类型，不读数值 → 不构成泄漏。
    """
    return X.select_dtypes(exclude="number").columns.tolist()


def build_preprocessor(X, derived_features: tuple = (), encode: bool = True,
                       target_encode: tuple = ()) -> Pipeline:
    """把"（可选）派生特征 → 填充 → （可选）目标编码 → （可选）编码"串成流水线。

    只使用 X 的【列名/类型】来决定各步处理哪些列（不读数值），因此不构成泄漏。
    真正的统计量（中位数、众数、类别清单、目标均值）都在 `fit` 时才学习。

    derived_features：要启用的派生特征名（默认空 = 基线版，用于 A/B 对比）。
    encode=False：**跳过编码**，只保留填充 → 给 CatBoost 这类
        "能自己处理原始类别列"的模型用（它需要看到原始字符串，而不是 One-Hot 后的 0/1）。
        ⚠️ 此时必须把类别列名通过模型的 `cat_features=...` 告诉它，否则模型会报错。
    target_encode：对哪些类别列做**目标编码**（OOF + 平滑，见 `OofTargetEncoder`）。
        这些列会在填充之后被换成数值，因此编码阶段会当作**数值透传**（不再 One-Hot）。
    """
    steps = []
    if derived_features:
        derive = AddDerivedFeatures(features=list(derived_features))
        # 派生列只会【新增】列名：构造 imputer/encoder 时把新列名并入"列清单"
        # （这里只看列名与类型，不读任何数值 → 无泄漏）
        col_names = X.columns.tolist() + list(derived_features)
        X_for_layout = X.reindex(columns=col_names)
        steps.append(("derive", derive))
    else:
        X_for_layout = X

    steps.append(("impute", build_imputer(X_for_layout)))

    if target_encode:
        steps.append(("target_encode",
                      OofTargetEncoder(columns=tuple(target_encode))))

    if encode:
        if target_encode:
            # 构造"布局"副本：把目标编码列**假装成数值**，让编码器把它们当数值透传，
            # 而不是再 One-Hot 一遍（否则会重复表达同一信息）。
            # ⚠️ 这里只改 dtype、**不读任何数值** —— ColumnTransformer 只按列名与类型派活。
            X_layout = X_for_layout.copy()
            for c in target_encode:
                if c in X_layout.columns:
                    X_layout[c] = pd.Series(0.0, index=X_layout.index)
        else:
            X_layout = X_for_layout
        steps.append(("encode", build_encoder(X_layout)))
    return Pipeline(steps)


def make_pipeline(model, X, derived_features: tuple = (),
                  encode: bool = True, target_encode: tuple = ()) -> Pipeline:
    """完整流水线：预处理 + 模型。**直接丢进 CV 即可**。

    derived_features：要启用的派生特征（默认空 = 基线版），用于 A/B 对比。
    encode=False：跳过 One-Hot/Ordinal 编码（见 `build_preprocessor`）。
    target_encode：对哪些类别列做目标编码（OOF + 平滑）。
    """
    return Pipeline(
        [
            ("prep", build_preprocessor(
                X, derived_features=derived_features,
                encode=encode, target_encode=target_encode,
            )),
            ("model", model),
        ]
    )
