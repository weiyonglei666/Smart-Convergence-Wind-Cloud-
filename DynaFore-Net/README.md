# DynaFore-Net

面向风电与光伏的动态时序表征与机理约束超短期功率预测网络。本目录从当前正式分场站实验中提取模型、训练与评估实现，提供独立 Python 包与命令行入口。

**本项目不包含数据集、预处理缓存、训练权重或场站预测结果。数据获取请联系 [ts26170007a31@cumt.edu.cn](mailto:ts26170007a31@cumt.edu.cn)。**

English overview: [README_EN.md](README_EN.md)

## 项目范围

- 独立风电/光伏模型：PMD-PVA-TRA、NSAN、自适应频谱、多尺度融合、未来气象交叉注意力与机理引导修正。
- AutoDA 任务适配与细粒度虚拟相位重构接入训练循环；增强策略和预测器联合优化。
- 光伏辐照/昼夜掩码及风光容量边界在模型输出头内执行。
- 每个场站独立训练，固定预测未来 **16 个 15 min 点，即 4 h**。
- 四组消融：`noPMD-PVA-TRA`、`noNSAN`、`noAutoDA`、`noguided correction`。
- 主指标：MSE、MAE、nMAE、nRMSE、ACC，另输出 RMSE、R²、爬坡事件 F1 等。

这是 DynaFore-Net 的独立整理包，不包含 XGBoost、PatchTST、iTransformer、Informer、TimeMixer、SimDiff 的基线仓库，也不包含此前的实验结果或论文图片。

## 安装

使用 Python 3.10 或更高版本，并按运行设备安装合适的 PyTorch。随后在项目根目录执行：

```bash
python -m pip install -e .
python -m dynafore_net --help
```

也可使用等价入口 `dynafore`。无需安装时可设置 `PYTHONPATH=src` 后运行 `python -m dynafore_net`。

## 数据准备

原始 CSV 由用户自行获取并保存在项目外部目录。文件名与字段要求见 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。容量必须来自场站资料或用户提供的 JSON，不从测试集功率反推。

```bash
dynafore prepare --raw-dir /path/to/private_csvs --output outputs
```

如果没有场站容量元数据 CSV，可传入 `--capacity-json /path/to/capacities.json`，文件结构为 `{ "场站ID": 容量MW }`。不要将该私人文件提交到仓库。

## 分场站训练

默认使用所有已准备场站、完整有效窗口与最多 20 epoch。每个场站单独拟合归一化统计量和模型。

```bash
dynafore train --output outputs --config configs/default.json --device cpu
```

仅选择指定场站：

```bash
dynafore train --output outputs --station FD0003 --device cuda:0
```

从原始数据一键准备并训练：

```bash
dynafore run --raw-dir /path/to/private_csvs --output outputs --device cuda:0
```

`--device` 默认是 `cpu`；GPU 需显式指定。已有缓存或模型默认不覆盖；需要重新运行时，请使用新输出目录，或明确传入 `--overwrite`。

## 消融实验

使用新的输出目录运行完整模型及四组消融：

```bash
dynafore run --raw-dir /path/to/private_csvs --output outputs_ablations \
  --variants full noPMD-PVA-TRA noNSAN noAutoDA "noguided correction" \
  --device cuda:0
```

`noAutoDA` 同时关闭其附属细粒度重构；`noPMD-PVA-TRA` 关闭局部均值解耦、PVA 和 TRA 的均值恢复，保留基础时间自注意力。`noguided correction` 关闭机理残差修正，保留输出物理边界与光伏昼夜掩码。

## 评估与结果

从最佳检查点重新计算测试结果，不重新拟合标准化统计量：

```bash
dynafore evaluate --output outputs --device cpu
dynafore summarize --output outputs
```

评估所有消融时须同样传入 `--variants full noPMD-PVA-TRA noNSAN noAutoDA "noguided correction"`。输出包括：

- `runs/<station>/<variant>/checkpoint.pt`：最佳模型、训练统计量、配置和对应增强参数；不是断点续训接口。
- `training_log.csv`、`metrics_val.json`、`metrics_test.json` 与逐起点预测误差。
- `results_by_station.csv`：各场站五个主指标及辅助指标。
- `results_variant_mean.csv`、`results_kind_variant_mean.csv`：非加权场站均值。
- `1st_count.csv`：逐场站最优指标计数；MSE/MAE/nMAE/nRMSE 越低越好，ACC 越高越好，并列最优各计一次。只统计输出目录中已存在的结果，不应把缺失消融的场站与完整比较混用。

这些文件仅在用户自行运行时生成，均不随源码分发。

## 默认参数与评价口径

历史窗口 288 点（72 h），预测长度 16；patch 长度/步长均为 16；隐藏维度 32；4 个注意力头；1 层 TRA；dropout 0.1；batch size 256。AdamW 初始学习率 `1e-3`，weight decay `1e-4`，梯度范数上限 1。

按有效预测起点顺序 70%/15%/15% 划分训练/验证/测试，清除标签跨越下一集合边界的窗口。仅训练窗口拟合标准化统计量。最多训练 20 epoch，至少 6 epoch 后允许早停；验证 ACC@16 连续 4 个 epoch 未改善则停止。ReduceLROnPlateau 按验证 ACC@16 调节学习率，factor 0.5、patience 2、最低学习率 `1e-5`。

**未来气象沿用当前实验的实测气象设置**：DynaFore-Net 使用预测区间对应的测风塔/气象站实测序列。这不是预测发布时刻已知的 NWP，不应把此条件下的测试精度描述为实际在线预报精度。历史气象只进行最多 4 步的前向填补，不使用后向填补或未来插值。详见 [实现与复现说明](docs/IMPLEMENTATION.md)。

## 测试与发布检查

测试只在内存及临时目录生成合成数据，不读取真实场站数据，不启动正式 20 epoch 实验。

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

项目结构：

```text
configs/             默认参数
src/dynafore_net/     模型、增强、数据处理、训练、评估、CLI
tests/               合成数据与模型行为测试
scripts/             无数据发布检查及打包工具
docs/                实现说明、来源记录与验证记录
```

## 使用条款与联系

依照项目负责人的选择，本整理包保留原研究使用条款，**不改为 MIT 或其他标准开源许可证**。具体授权范围见 [LICENSE](LICENSE)；公开源码整理不等于无条件再分发或商业使用许可。

数据获取、研究合作及授权事宜：**ts26170007a31@cumt.edu.cn**。方法来源与引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
