"""验证 sklearn 的三个关键概念：fit 原地修改 / clone / Pipeline。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_sklearn_concepts.py

要点：
- fit 是"原地修改模型对象"：同一个对象 fit 后从"没有参数"变成"有参数"
- clone 造出"同配置、但未训练"的副本：超参数保留、学到的参数清空
- Pipeline 把多步打包成一个对象：named_steps 取步骤、set_params("步骤__参数") 改参数
- 顺带给 Ridge 的正则化留个证据：alpha 越大，系数被压得越小（y=2x 时 alpha=1 学到 1.333 而非 2）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])  # 真实关系 y = 2x


def main() -> None:
    # ---- 0) Ridge 的正则化：alpha 影响学到的系数 ----
    print("=== 0) 同一个 y=2x，不同模型的系数 ===")
    for name, mdl in [
        ("Ridge(alpha=1.0)", Ridge(alpha=1.0)),
        ("Ridge(alpha=0.0)", Ridge(alpha=0.0)),  # alpha=0 等价普通最小二乘
        ("LinearRegression", LinearRegression()),
    ]:
        mdl.fit(X, y)
        print(f"  {name:18s} coef_={mdl.coef_} intercept_={mdl.intercept_:.4f}")

    # ---- 1) fit 原地修改对象：对象没换，状态变了 ----
    print("\n=== 1) fit 原地修改（对象没变，状态变了）===")
    m = Ridge(alpha=1.0)
    before_id = id(m)
    print(f"  fit 前 有 coef_ 吗: {hasattr(m, 'coef_')}")
    m.fit(X, y)
    print(f"  fit 后 有 coef_ 吗: {hasattr(m, 'coef_')}  coef_={m.coef_}")
    print(f"  fit 前后是同一个对象吗: {before_id == id(m)}")

    # ---- 2) clone：复制配置、清空训练状态 ----
    print("\n=== 2) clone ===")
    c = clone(m)
    print(f"  原模型: coef_ 存在={hasattr(m, 'coef_')}  alpha={m.alpha}")
    print(f"  副本  : coef_ 存在={hasattr(c, 'coef_')}  alpha={c.alpha}")

    # ---- 3) Pipeline：把多步打包成一个对象 ----
    print("\n=== 3) Pipeline ===")
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    yy = np.array([1.0, 2.0, 3.0, 4.0])
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", Ridge()),
        ]
    )
    print(f"  步骤名 (named_steps): {list(pipe.named_steps)}")
    pipe.fit(df, yy)
    print(f"  named_steps['model'].alpha = {pipe.named_steps['model'].alpha}")
    pipe.set_params(model__alpha=10.0)
    print(f"  set_params(model__alpha=10) 后 = {pipe.named_steps['model'].alpha}")
    pc = clone(pipe)
    print(f"  clone(pipe) 步骤保留: {list(pc.named_steps)}  alpha={pc.named_steps['model'].alpha}")
    print(f"  预测: {np.round(pipe.predict(df), 3)}")


if __name__ == "__main__":
    main()
