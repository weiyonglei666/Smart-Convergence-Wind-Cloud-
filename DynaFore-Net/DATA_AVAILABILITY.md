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

## 字段与单位

| 文件 | 必需/使用字段 | 单位及说明 |
|---|---|---|
| 功率 | `DATE_TIME`, `ACTIVE_POWER` | 功率 MW；时间统一为场站当地时间，无夏令时跳变或混合时区 |
| 风电气象 | `DATE_TIME`，`WS*`、`WD*` 等数值列 | 风速 m/s，风向度；按代码优先顺序选择最多 10 个气象通道 |
| 光伏气象 | `DATE_TIME`，`GHI`/`GTI`（可用 `DNI` 作为第二派生辐照通道），其他气象列 | 辐照 W/m²；昼夜判据取两个辐照通道最大值是否大于 20 W/m² |
| 风电容量元数据 | `FARM_ID`, `CAP` | 额定容量 MW；文件 `zbase_wind_farm_info_encryption.csv` |
| 光伏容量元数据 | `FARM_ID`，`CAP_REPORT`/`CAP_AC`/`CAP_DC` | 容量 MW；按此顺序取首个有效正值；文件 `zbase_photovoltaic_farm_info.csv` |

时间推荐使用 `YYYY-MM-DD HH:MM:SS`，原实验的日/月/年格式也受支持。同一文件保持一致格式。重复时间记录取数值均值；采样对齐到 15 min 网格。功率缺失不填补，气象最多前向填补 4 步，剩余缺失使相关窗口无效。

无容量元数据时，使用私有 JSON 以 `--capacity-json` 传入站号到容量 MW 的映射。容量缺失会报错，不从测试集估算容量。

自动发现的站号为文件名中 `<station>` 部分，可通过 `--station` 选择。不要求固定 11 个场站；各场站始终独立建模与评估。默认历史通道是功率 + 最多 10 个气象通道 + 6 个时间周期通道；标准 10 气象通道时历史/未来维度分别为 17/18。

## 未来气象的含义

本整理版本沿用作者指定的实测未来气象实验设置。模型目标和未来气象窗口均为 `[p:p+16]`，历史窗口为 `[p-lookback:p]`。缓存中的 `origin_time` 是第一个目标点时间，最后一个历史观测比它早 15 min。

内部 `nwp__pl925__u`、`nwp__pl925__v` 是继承的通道名称；在当前数据管线中，风电对应由实测风速/风向派生的两个分量，光伏对应两个辐照通道，**不表示真的读取了 NWP 或 925 hPa 数据**。

## 访问与再分发

数据访问与再分发需以数据提供者的授权为准。邮件联系不代表自动获得数据使用或再分发许可。未经授权不要将原始数据、容量元数据、检查点或预测结果提交到公开仓库。
