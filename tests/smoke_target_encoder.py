"""冒烟：验证 `OofTargetEncoder`（目标编码）真的做了 OOF + 平滑（**不算分数**）。

运行方式（项目根目录）：
    .\\.venv\\Scripts\\python.exe tests\\smoke_target_encoder.py

为什么要专门测这个？因为目标编码是**最容易悄悄泄漏**的特征工程手段，
而泄漏的表现是"分数突然变好"—— 不报错、很好看、但全是假的。

本测试用三条**可精确检验**的性质来判定（不靠"分数看起来对不对"）：

  ① **单样本类别的 OOF 编码必须恰好等于全局均值（prior）**
     —— 该类别只有 1 行，做 OOF 时它自己被留出 → 训练部分里这个类别 0 条
     → 平滑公式 (0*? + m*prior)/(0+m) = prior。
     如果实现里没做 OOF，这个值会变成"它自己的 y"（几乎必然 ≠ prior）。

  ② **训练行的 OOF 编码 ≠ transform 的编码**
     —— 两条路径本就该不同（前者是 OOF，后者用全量训练数据）。
     如果两者一模一样，说明 fit_transform 根本没做 K 折。

  ③ **平滑公式可手算核对**
     —— 取一个类别，用 (n*mean + m*prior)/(n+m) 手算，应与 transform 输出一致。

  ④ 端到端：装进 Pipeline 能 fit / predict，无 NaN，列名与顺序不变。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_prices.preprocess import OofTargetEncoder, make_pipeline  # noqa: E402

SMOOTHING = 10.0


def main() -> None:
    # —— 玩具数据：类别 A 只出现 1 次（关键样本），其余类别各 5 次 ——
    cats = ["A"] + ["B"] * 5 + ["C"] * 5 + ["D"] * 5
    rng = np.random.default_rng(0)
    base = {"A": 1.0, "B": 5.0, "C": 9.0, "D": 13.0}
    y = np.array([base[c] for c in cats], dtype=float) + rng.normal(0, 0.01, len(cats))
    X = pd.DataFrame({"City": cats, "Num": np.arange(len(cats), dtype=float)})

    enc = OofTargetEncoder(columns=("City",), n_splits=5, smoothing=SMOOTHING, seed=42)
    X_oof = enc.fit_transform(X, y)          # 训练路径（OOF）
    X_all = enc.transform(X)                 # 推理路径（用全量训练统计量）

    prior = float(np.mean(y))
    print(f"全局均值 prior = {prior:.5f}（= transform 时的回退值）")
    print(f"smoothing m = {SMOOTHING}（先验相当于 {SMOOTHING:.0f} 条虚拟样本）\n")

    # ---- ① 单样本类别：OOF 值必须 == prior ----
    a_oof = float(X_oof.loc[0, "City"])
    ok1 = bool(np.isclose(a_oof, prior, atol=1e-9))
    print(f"① 单样本类别 A 的 OOF 编码 = {a_oof:.6f}")
    print(f"   应恰好等于 prior {prior:.6f} → {'✅ 通过（OOF 生效）' if ok1 else '❌ 失败（疑似没做 OOF！）'}")

    # ---- ② 两条路径必须不同 ----
    x_oof = X_oof["City"].to_numpy()
    x_all = X_all["City"].to_numpy()
    ok2 = bool(not np.allclose(x_oof, x_all))
    print(f"\n② 训练路径(OOF) 与 推理路径(全量) 是否不同 → "
          f"{'✅ 通过' if ok2 else '❌ 失败（fit_transform 没做 K 折）'}")
    print(f"   例：类别 A 的 transform 值 = {x_all[0]:.6f}（≠ prior，因为它用到了自己那一行）")
    print(f"       类别 B 的 OOF = {x_oof[1]:.5f}，transform = {x_all[1]:.5f}")

    # ---- ③ 平滑公式手算核对 ----
    n_b = int(np.sum(np.array(cats) == "B"))
    mean_b = float(np.mean(y[np.array(cats) == "B"]))
    expect_b = (n_b * mean_b + SMOOTHING * prior) / (n_b + SMOOTHING)
    ok3 = bool(np.isclose(x_all[1], expect_b, atol=1e-9))
    print(f"\n③ 手算核对类别 B：")
    print(f"   n={n_b}, 类别均值={mean_b:.5f}, prior={prior:.5f}")
    print(f"   (n*mean + m*prior)/(n+m) = {expect_b:.6f}")
    print(f"   实现输出                = {x_all[1]:.6f}  → {'✅ 一致' if ok3 else '❌ 不一致'}")

    # ---- ④ 端到端：装进 Pipeline ----
    from sklearn.linear_model import Ridge   # 局部导入，避免污染上文阅读

    pipe = make_pipeline(Ridge(), X, target_encode=("City",))
    pipe.fit(X, y)
    pred = pipe.predict(X)
    Xt = pipe.named_steps["prep"].transform(X)
    ok4 = bool(
        np.all(np.isfinite(pred))
        and not Xt.isna().any().any()
        and list(Xt.columns) == ["City", "Num"]
    )
    print(f"\n④ 端到端 Pipeline：fit/predict 成功")
    print(f"   变换后列 = {list(Xt.columns)}（顺序不变）、无 NaN、预测全有限 → "
          f"{'✅ 通过' if ok4 else '❌ 失败'}")

    if not all([ok1, ok2, ok3, ok4]):
        raise SystemExit("\n❌ 有用例未通过，先别拿它做实验")


if __name__ == "__main__":
    main()
