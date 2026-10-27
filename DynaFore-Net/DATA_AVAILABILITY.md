# 数据获取与输入格式 / Data availability

**数据集不随本项目发布。请联系 ts26170007a31@cumt.edu.cn 获取数据或咨询访问条件。**

**Datasets are not distributed with this repository. Please contact ts26170007a31@cumt.edu.cn for dataset access and access conditions.**

源码包也不包含场站容量表、预处理数组、检查点、预测结果或真实数据片段。测试使用运行时临时生成的合成数据。

## 私人数据目录

运行时通过 `--raw-dir` 指定目录，支持以下原实验文件命名：

```text
zdata_<station>_wind_power_real_15min.csv
zdata_<station>_wind_tower_15min.csv
zdata_<station>_pv_power_real_15min.csv
zdata_<station>_qxz_15min.csv
```

气象文件不存在时，可使用同名的 `_15min_history.csv` 文件。每个场站应提供与其类型匹配的功率和气象文件。

## 未来 NWP 文件

文档规定的未来条件是 ECMWF 类 NWP，而不是预测区间的实测气象。`prepare` / `run` 默认使用 `--future-source nwp`，此时必须提供 `--nwp-dir`，目录中按场站放置：

```text
zdata_<station>_nwp.csv
# 或 zdata_<station>_nwp_15min.csv / <station>_nwp.csv
```

文件为 UTF-8 CSV，至少包含：

| 字段 | 说明 |
|---|---|
| `ISSUE_TIME` | 起报时间，原始 UTC；可用 `Z` 或 `+00:00` |
| `DATE_TIME` | 预报有效时间，原始 UTC；可含多个起报批次 |
| 物理变量 | 风速/风向（风电场站必需）、温度、湿度、气压、云量、降水、GHI/GTI（光伏必需）等 |

处理规则与文档一致：

1. `ISSUE_TIME` 和 `DATE_TIME` 转为北京时间（UTC+8）后与功率 15 min 时间轴对齐；
2. 只允许使用 `ISSUE_TIME <= 预测起点` 的批次，避免未来信息泄漏；
3. 原生 3 h（或其他步长）序列在每个起报批次内线性插值为 15 min；
4. 空间匹配由平台 NWP 提取脚本按场站经纬度双线性插值后裁剪，训练侧读取场站级序列；
5. 未来条件固定为 18 维：2 个风分量 + 10 个 NWP 物理变量 + 6 个时间周期特征。

若显式设置 `--future-source measured`，则使用预测区间的实测气象作为**消融设置**；该模式会在 `meta.json` 中标记为 `measured_future`，不能描述为在线 NWP 精度。

## 字段与单位

| 文件 | 必需/使用字段 | 单位及说明 |
|---|---|---|
| 功率 | `DATE_TIME`, `ACTIVE_POWER` | 功率 MW；时间统一为场站当地时间，无夏令时跳变或混合时区 |
| 风电气象 | `DATE_TIME`，`WS*`、`WD*` 等数值列 | 风速 m/s，风向度；按代码优先顺序选择恰好 10 个气象通道 |
| 光伏气象 | `DATE_TIME`，`GHI`/`GTI`（可用 `DNI` 作为第二派生辐照通道），其他气象列 | 辐照 W/m²；昼夜判据取两个辐照通道最大值是否大于 20 W/m² |
| 风电容量元数据 | `FARM_ID`, `CAP` | 额定容量 MW；文件 `zbase_wind_farm_info_encryption.csv` |
| 光伏容量元数据 | `FARM_ID`，`CAP_REPORT`/`CAP_AC`/`CAP_DC` | 容量 MW；按此顺序取首个有效正值；文件 `zbase_photovoltaic_farm_info.csv` |

时间推荐使用 `YYYY-MM-DD HH:MM:SS`，原实验的日/月/年格式也受支持。同一文件保持一致格式。重复时间记录取数值均值；采样对齐到 15 min 网格。功率缺失不填补，气象最多前向填补 4 步，剩余缺失使相关窗口无效。

无容量元数据时，使用私有 JSON 以 `--capacity-json` 传入站号到容量 MW 的映射。容量缺失会报错，不从测试集估算容量。

自动发现的站号为文件名中 `<station>` 部分，可通过 `--station` 选择。不要求固定 11 个场站；各场站始终独立建模与评估。历史/未来维度固定为 17/18：功率 + 10 个气象通道 + 6 个时间周期通道。风电场站历史窗口 288 步（72 h），光伏 96 步（24 h）。

## 未来气象的含义

`meta.json` 的 `future_weather_source` 明确区分 `nwp` 与 `measured_future`。历史窗口为 `[p-lookback:p]`，目标窗口为 `[p:p+16]`；缓存中的 `origin_time` 是第一个目标点时间，最后一个历史观测比它早 15 min。

内部通道统一命名为 `future__wind_u`、`future__wind_v`、`nwp__<variable>`；不再用 `nwp__pl925__u` 表示实测派生量。旧检查点中的旧通道名仍可被模型读取，但新准备的数据使用上述规范名称。

## 访问与再分发

数据访问与再分发需以数据提供者的授权为准。邮件联系不代表自动获得数据使用或再分发许可。未经授权不要将原始数据、容量元数据、检查点或预测结果提交到公开仓库。
