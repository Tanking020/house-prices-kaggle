"""读取原始数据。

约定：data/raw/ 只读，本模块只负责"读进来"，不做任何修改。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 本文件位于 <repo>/src/house_prices/data.py
# parents[0]=house_prices, parents[1]=src, parents[2]=仓库根目录
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取原始训练集 / 测试集。

    Returns
    -------
    (train, test)
        train: 1460 x 81，含 SalePrice
        test : 1459 x 80，不含 SalePrice
    """
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    return train, test


def get_xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """把 train 拆成 (X, y)。

    注意：Id 是行号、不是特征，必须排除。
    y 返回的是**原始尺度**的 SalePrice（log 变换在评估/训练时再做）。
    """
    X = train.drop(columns=["Id", "SalePrice"])
    y = train["SalePrice"]
    return X, y
