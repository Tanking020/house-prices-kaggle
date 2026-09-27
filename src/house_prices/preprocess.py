"""预处理：把 EDA 的结论变成可复用的 sklearn 变换器。

⚠️ 铁律：这里的变换全都必须放进 Pipeline，
   只在【训练集】上 fit，再 transform 验证集 / 测试集 —— 否则数据泄漏。
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
