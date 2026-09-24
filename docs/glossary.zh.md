# House Prices 数据字典 · 中英对照速查

> 用途：`data/raw/data_description.txt` 的中文辅助手册。
> 原则：**英文是代码里的列名/取值，不能改；中文只做理解辅助**。
> 建议先看「一、通用评级代码」，再按需查「二、字段速查」。

---

## 一、通用评级代码（出现频率最高，务必先记）

这套代码在 `ExterQual / ExterCond / BsmtQual / BsmtCond / HeatingQC /
KitchenQual / FireplaceQu / GarageQual / GarageCond / PoolQC` 里反复出现：

| 代码 | 英文 | 中文 | 含义 |
|------|------|------|------|
| `Ex` | Excellent | 优秀 | 最高档 |
| `Gd` | Good | 良好 | |
| `TA` | Typical / Average | 典型 / 一般 | **默认档，出现最多** |
| `Fa` | Fair | 较差 | |
| `Po` | Poor | 很差 | 最低档 |
| `NA` | — | 无 | ⚠️ 通常是**没有该设施**，不是数据缺失 |

**重要**：这是**有序类别**（`Ex > Gd > TA > Fa > Po`），
编码时应映射成数字保序，例如 `{Po:1, Fa:2, TA:3, Gd:4, Ex:5}`。

> ⚠️ 陷阱：`NA` 在不同字段含义不同——
> - `BsmtQual = NA` → 没有地下室
> - `FireplaceQu = NA` → 没有壁炉
> - `GarageType = NA` → 没有车库
> - 而 `LotFrontage` 的缺失是**真的缺失**（要用统计量填补）
> 判断方法：**看数据字典里该字段是否显式列出了 `NA` 这个取值**。

---

## 二、79 个字段速查（按语义分 5 类）

### A. 面积 / 尺寸类（数值，注意偏态和离群）

| 字段 | 中文 | 单位/说明 |
|------|------|-----------|
| `LotFrontage` | 临街正面长度 | 英尺（缺失多） |
| `LotArea` | 地块面积 | 平方英尺 |
| `MasVnrArea` | 砌体饰面面积 | 平方英尺 |
| `BsmtFinSF1` | 地下室装修面积-类型1 | 平方英尺 |
| `BsmtFinSF2` | 地下室装修面积-类型2 | 平方英尺 |
| `BsmtUnfSF` | 地下室未装修面积 | 平方英尺 |
| `TotalBsmtSF` | 地下室总面积 | 平方英尺 |
| `1stFlrSF` | 一层面积 | 平方英尺 |
| `2ndFlrSF` | 二层面积 | 平方英尺 |
| `LowQualFinSF` | 低质量装修面积（全楼层） | 平方英尺 |
| `GrLivArea` | **地上居住面积** | 平方英尺 ⭐ 最重要 |
| `GarageArea` | 车库面积 | 平方英尺 |
| `WoodDeckSF` | 木质露台面积 | 平方英尺 |
| `OpenPorchSF` | 开放式门廊面积 | 平方英尺 |
| `EnclosedPorch` | 封闭式门廊面积 | 平方英尺 |
| `3SsnPorch` | 三季门廊面积 | 平方英尺 |
| `ScreenPorch` | 纱窗门廊面积 | 平方英尺 |
| `PoolArea` | 泳池面积 | 平方英尺（大多为 0） |
| `MiscVal` | 杂项设施价值 | 美元 |

### B. 质量 / 评级类

| 字段 | 中文 | 取值 |
|------|------|------|
| `OverallQual` | **整体材料与工艺评分** | 1–10 整数 ⭐ 最重要 |
| `OverallCond` | 整体状况评分 | 1–10 整数 |
| `ExterQual` | 外部材料质量 | 通用评级 |
| `ExterCond` | 外部材料现状 | 通用评级 |
| `BsmtQual` | 地下室高度/质量 | 通用评级 + `NA`(无地下室) |
| `BsmtCond` | 地下室总体状况 | 通用评级 + `NA` |
| `BsmtExposure` | 地下室采光/出口 | `Gd/Av/Mn/No/NA` |
| `BsmtFinType1` | 地下室装修等级-1 | `GLQ/ALQ/BLQ/Rec/LwQ/Unf/NA` |
| `BsmtFinType2` | 地下室装修等级-2 | 同上 |
| `HeatingQC` | 供暖质量 | 通用评级 |
| `KitchenQual` | 厨房质量 | 通用评级 ⭐ |
| `FireplaceQu` | 壁炉质量 | 通用评级 + `NA`(无壁炉) |
| `GarageQual` | 车库质量 | 通用评级 + `NA` |
| `GarageCond` | 车库状况 | 通用评级 + `NA` |
| `GarageFinish` | 车库内饰 | `Fin/RFn/Unf/NA` |
| `PoolQC` | 泳池质量 | 通用评级 + `NA`（1453/1460 缺失） |
| `Functional` | 功能性 | `Typ/Min1/Min2/Mod/Maj1/Maj2/Sev/Sal` |

### C. 时间类

| 字段 | 中文 | 说明 |
|------|------|------|
| `YearBuilt` | 建造年份 | → 可转成"房龄" |
| `YearRemodAdd` | 翻新年份 | 无翻新 = 建造年份 |
| `GarageYrBlt` | 车库建造年份 | 无车库时缺失 |
| `MoSold` | 售出月份 | 1–12 |
| `YrSold` | 售出年份 | 2006–2010 |

### D. 位置 / 分类类（名义变量，用 One-Hot）

| 字段 | 中文 | 备注 |
|------|------|------|
| `MSSubClass` | 住宅类型代码 | 数字表示**类别**，别当数值 ⚠️ |
| `MSZoning` | 分区类别 | 住宅/商业/工业等 |
| `Street` | 道路类型 | Grvl/Pave |
| `Alley` | 巷道类型 | 约 1369 缺失 = 无巷道 |
| `LotShape` | 地块形状 | Reg/IR1/IR2/IR3（**有序**） |
| `LandContour` | 地形 | Lvl/Bnk/HLS/Low |
| `Utilities` | 公用设施 | 几乎全 AllPub（可丢弃） |
| `LotConfig` | 地块布局 | Inside/Corner/CulDSac/FR2/FR3 |
| `LandSlope` | 坡度 | Gtl/Mod/Sev（**有序**） |
| `Neighborhood` | **社区** | 25 个取值 ⭐ 重要 |
| `Condition1` | 邻近条件-1 | 临街/铁路/公园等 |
| `Condition2` | 邻近条件-2 | 同上（大多 Norm） |
| `BldgType` | 建筑类型 | 1Fam/2FmCon/Duplx/TwnhsE/TwnhsI |
| `HouseStyle` | 房屋风格 | 1Story/2Story/SFoyer/SLvl 等 |
| `RoofStyle` | 屋顶风格 | Flat/Gable/Gambrel/Hip/Mansard/Shed |
| `RoofMatl` | 屋顶材料 | 多为 CompShg |
| `Exterior1st` | 外墙材料-1 | 16 种 |
| `Exterior2nd` | 外墙材料-2 | 16 种 |
| `MasVnrType` | 砌体饰面类型 | BrkCmn/BrkFace/CBlock/None/Stone |
| `Foundation` | 地基类型 | BrkTil/CBlock/PConc/Slab/Stone/Wood |
| `Heating` | 供暖类型 | 多为 GasA |
| `CentralAir` | 中央空调 | Y/N |
| `Electrical` | 电气系统 | SBrkr/FuseA/FuseF/FuseP/Mix |
| `GarageType` | 车库位置 | Attchd/Detchd/BuiltIn/Basment/CarPort/2Types/NA |
| `PavedDrive` | 铺装车道 | Y/P/N（**有序**） |
| `Fence` | 围栏 | GdPrv/MnPrv/GdWo/MnWw/NA |
| `MiscFeature` | 杂项设施 | Elev/Gar2/Othr/Shed/TenC/NA |
| `SaleType` | 销售类型 | WD/CWD/VWD/New/COD/Con/ConLw/ConLI/ConLD/Oth |
| `SaleCondition` | 销售条件 | Normal/Abnorml/AdjLand/Alloca/Family/Partial |

### E. 设施 / 计数类

| 字段 | 中文 |
|------|------|
| `BsmtFullBath` | 地下室全功能卫浴数 |
| `BsmtHalfBath` | 地下室半功能卫浴数 |
| `FullBath` | 地上全功能卫浴数 |
| `HalfBath` | 地上半功能卫浴数 |
| `BedroomAbvGr` | 地上卧室数（不含地下室） |
| `KitchenAbvGr` | 地上厨房数 |
| `TotRmsAbvGrd` | 地上总房间数（不含卫浴） |
| `Fireplaces` | 壁炉数量 |
| `GarageCars` | 车库容量（车位数） |

---

## 三、易混 / 重要取值代码表

### 1. `Neighborhood` 社区（25 个）
| 代码 | 含义 | 代码 | 含义 |
|------|------|------|------|
| `Blmngtn` | Bloomington Heights | `NoRidge` | Northridge |
| `Blueste` | Bluestem | `NPkVill` | Northpark Villa |
| `BrDale` | Briardale | `NridgHt` | Northridge Heights |
| `BrkSide` | Brookside | `NWAmes` | Northwest Ames |
| `ClearCr` | Clear Creek | `OldTown` | Old Town |
| `CollgCr` | College Creek | `SWISU` | South & West of ISU |
| `Crawfor` | Crawford | `Sawyer` | Sawyer |
| `Edwards` | Edwards | `SawyerW` | Sawyer West |
| `Gilbert` | Gilbert | `Somerst` | Somerset |
| `IDOTRR` | Iowa DOT and Rail Road | `StoneBr` | Stone Brook |
| `MeadowV` | Meadow Village | `Timber` | Timberland |
| `Mitchel` | Mitchell | `Veenker` | Veenker |
| `Names` | North Ames | | |

### 2. 有序类别（务必保序编码）
- 通用评级：`Po(1) < Fa(2) < TA(3) < Gd(4) < Ex(5)`
- `LotShape`：`Reg(3) > IR1(2) > IR2(1) > IR3(0)`
- `LandSlope`：`Gtl(2) > Mod(1) > Sev(0)`
- `PavedDrive`：`Y(2) > P(1) > N(0)`
- `BsmtExposure`：`Gd > Av > Mn > No`
- `BsmtFinType`：`GLQ > ALQ > BLQ > Rec > LwQ > Unf`
- `Functional`：`Typ > Min1 > Min2 > Mod > Maj1 > Maj2 > Sev > Sal`
- `GarageFinish`：`Fin > RFn > Unf`
- `Fence`：`GdPrv > MnPrv > GdWo > MnWw`

### 3. `MSSubClass`（数字其实是类别 ⚠️）
代码含义见 `data_description.txt` 开头，例如
`20`=1层1946年后, `60`=2层1946年后, `90`=双拼 等。
**建模时应视为分类变量**（One-Hot 或转字符串）。

---

## 四、翻译工作流建议

1. **主力**：本文件 + `data_description.txt` 原文对照看
2. **临时查词**：VS Code 装 "Comment Translate" 划词插件，或 DeepL / 有道
3. **批量翻译**：让 AI 把某段原文译成中英对照
4. **别做**：不要把英文列名改成中文——代码会崩
5. **进阶**：生词积累够了，直接读英文原文，这是 AI 工程师的必备能力
