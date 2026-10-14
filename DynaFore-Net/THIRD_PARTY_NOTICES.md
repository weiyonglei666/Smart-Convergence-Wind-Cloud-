# 方法来源与第三方说明

本包保留当前研究项目自己的 DynaFore-Net 实现，仅包含运行该模型所需的本地模块，不打包第三方数据、完整基线仓库或权重。

- **PMDformer**：patch-mean decoupling、PVA 与 TRA 思路参考 [PMDformer 作者仓库](https://github.com/aohu1105/PMDformer) 与其中论文信息。本包 `attention.py` 来自原研究项目的本地 PVA/TRA 实现，不是完整 PMDformer 基线。
- **AutoDA-Timeseries**：特征条件策略增强参考 [作者项目页面](https://netmanaiops.github.io/AutoDA-Timeseries/)。本包包含的是新能源任务适配实现，不声称复现作者全部操作与损失。
- **PyTorch、NumPy、pandas**：运行时依赖，通过包管理器安装；其许可证仍由各自项目维护，本源码包不重新授权这些依赖。

原项目的研究使用条款完整保留在 `LICENSE` 中。论文使用应按所用方法引用相关原始研究；本包没有虚构 DynaFore-Net 的 DOI、作者列表或已发表论文信息。
