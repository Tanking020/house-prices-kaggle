"""预处理：把 EDA 的结论变成可复用的 sklearn 变换器。

⚠️ 铁律：这里的变换全都必须放进 Pipeline，
   只在【训练集】上 fit，再 transform 验证集 / 测试集 —— 否则数据泄漏。
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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


def cat_feature_names(X) -> list[str]:
    """列出所有【类别型】列名（= 非数值列）。

    用途：有些库（如 CatBoost）不需要先做 One-Hot，而是**直接吃原始类别列**，
    此时要把这 43（或加上 MSSubClass 就 44）列的列名告诉它。
    只读列名/类型，不读数值 → 不构成泄漏。
    """
    return X.select_dtypes(exclude="number").columns.tolist()


def build_preprocessor(X, derived_features: tuple = (), encode: bool = True) -> Pipeline:
    """把"（可选）派生特征 → 填充 → （可选）编码"串成一条预处理流水线。

    只使用 X 的【列名/类型】来决定各步处理哪些列（不读数值），因此不构成泄漏。
    真正的统计量（中位数、众数、类别清单）都在 `fit` 时才学习。

    derived_features：要启用的派生特征名（默认空 = 基线版，用于 A/B 对比）。
    encode=False：**跳过编码**，只保留填充 → 给 CatBoost 这类
        "能自己处理原始类别列"的模型用（它需要看到原始字符串，而不是 One-Hot 后的 0/1）。
        ⚠️ 此时必须把类别列名通过模型的 `cat_features=...` 告诉它，否则模型会报错。
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
    if encode:
        steps.append(("encode", build_encoder(X_for_layout)))
    return Pipeline(steps)


def make_pipeline(model, X, derived_features: tuple = (),
                  encode: bool = True) -> Pipeline:
    """完整流水线：预处理 + 模型。**直接丢进 CV 即可**。

    derived_features：要启用的派生特征（默认空 = 基线版），用于 A/B 对比。
    encode=False：跳过 One-Hot/Ordinal 编码（见 `build_preprocessor`）。
    """
    return Pipeline(
        [
            ("prep", build_preprocessor(
                X, derived_features=derived_features, encode=encode
            )),
            ("model", model),
        ]
    )
