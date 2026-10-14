# Disco-RAG for Power-Domain Question Answering

电力领域 Disco-RAG 适配与评估代码，基于 [dongqi-me/Disco-RAG](https://github.com/dongqi-me/Disco-RAG) 的结构模块。本仓库不是上游官方实现；提供 OpenAI-compatible API 接入、Qwen3-Embedding-8B / FAISS 检索，以及五种方法的全量评估入口。

## 数据获取

**本仓库不包含数据集、知识库原文、模型权重、向量索引或实验生成答案。数据获取及使用许可事宜，请联系 [ts26170007a31@cumt.edu.cn](mailto:ts26170007a31@cumt.edu.cn)。**

第三方数据与模型仍受其各自许可约束，联系邮箱并不代表对所有上游资料拥有再分发权。收到数据后按 [数据格式说明](data/README.md) 放置；知识库与评测问题应分开，禁止将测试参考答案放入检索知识库。

## 方法与默认参数

| 方法 ID | 方法 | RST 树 | 修辞图 | 规划 |
|---|---|---|---|---|
| disco_full | Disco-RAG full | 是 | 是 | 是 |
| without_rst | w/o RST tree | 否 | 是 | 是 |
| without_graph | w/o rhetorical graph | 是 | 否 | 是 |
| without_planning | w/o planning | 是 | 是 | 否 |
| standard_rag | Standard RAG | 否 | 否 | 否 |

发布入口默认从第 0 条开始运行全部评测问题（`--start-index 0 --max-items 0`），不使用后 200 条子集。五组统一 top-k=10、temperature=0、最终生成预算=1024 token、英文输出；结构阶段单次预算为256 token。嵌入使用 Qwen3-Embedding-8B、last-token pooling、归一化向量，分块长度256 token、重叠0，通过FAISS归一化内积检索。

这里的参数是发布版运行配置，不代表已重新完成全量实验，也不保证复现此前聊天中的汇总数值。本仓库不提供未经原始运行记录核验的结果表。

## 安装

建议 Python 3.10+，CUDA 机器用于8B嵌入模型和可选的语义指标。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 可选：指标计算与测试
pip install -r requirements-metrics.txt
pip install -r requirements-dev.txt
```

根据本机 CUDA 环境安装适配的 PyTorch。生成模型由外部 OpenAI-compatible 服务提供，本仓库不负责开机自启或启动模型服务，也不需要为 API 生成安装 vLLM。保留的上游本地模型模块若单独使用，则需要自行安装 vLLM。

## 配置

```bash
cp .env.example .env
# 编辑 .env：设置实际服务模型名称、地址、API key、数据和模型路径
set -a
source .env
set +a
```

模型名称必须与 API `/v1/models` 返回的 ID 一致；`Qwen3.8-27B` 仅是此前服务的部署名。`.env` 不得提交。

## 构建知识库索引

先在本地准备嵌入模型和 UTF-8 TXT 知识文件：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_power_faiss.py \
  --corpus "$POWER_CORPUS_ROOT" \
  --model "$POWER_EMBEDDING_MODEL" \
  --output "$POWER_FAISS_INDEX" \
  --chunk-size 256 --batch-size 2
```

`CUDA_VISIBLE_DEVICES` 仅选择嵌入模型进程使用的卡，不控制远端生成服务的GPU。索引含源文本与pickle对象，不应提交；仅加载自己构建或可信来源的索引，pickle反序列化可能执行代码。

## 全量实验与断点续跑

```bash
python scripts/run_power_eval.py \
  --methods disco_full without_rst without_graph without_planning standard_rag \
  --start-index 0 --max-items 0 \
  --output-dir results/full_en
```

正式运行前可添加 `--dry-run` 检查问题数量和路径，不调用生成 API。默认五种方法顺序执行；成功且非空的已有答案会跳过，失败或空答案会重试。`Ctrl+C` 中断后使用同一配置、同一输出目录重运行即可。改变模型、参数或知识库时必须换新输出目录，避免混合不同实验。

`--max-items 0` 表示全量，而不是0个样本。可选子集参数仅供调试，不作为默认发布实验。生成耗时包括在线检索、结构解析、规划和生成；当前适配版在线解析RST，不应声称其ART排除了该阶段。

## 计算五项指标

```bash
python scripts/evaluate_saved_answers.py --stage basic \
  --input-dir results/full_en --output-dir results/full_metrics
python scripts/evaluate_saved_answers.py --stage bert \
  --input-dir results/full_en --output-dir results/full_metrics --device cuda:0
python scripts/evaluate_saved_answers.py --stage summac --summac-mode entailment \
  --input-dir results/full_en --output-dir results/full_metrics --device cuda:0
```

输出 `summary.csv` / `summary.json`，包含 Token F1、ROUGE-L F1、BERTScore-F1、SummaCZS 和 ART。BERTScore 使用 roberta-large 的第17层，IDF关闭，不做baseline rescaling。SummaC使用 vitc / sentence 粒度；`entailment` 为只使用蕴含分数的设置，`signed` 为包含矛盾分数的设置，应在论文中明确，不能混用。指标模型需另行下载。

摘要同时报告各指标有效样本数量；未完成的阶段不是全量评估。指标计算不重新生成答案，不计入ART。没有人工计时实验时，不将情景假设写成实测人工效率收益。

## 测试

```bash
python -m pytest tests -q
```

测试使用临时合成文本与mock客户端，不需要实际数据集或在线生成模型。单元测试不等于真实模型全量复现。

## 目录

```text
src/          上游结构、提示词、生成和检索模块
power_eval/   电力领域API适配、五种方法和数据读写
scripts/      索引构建、全量运行、已有答案评分
tests/        无真实数据的单元测试
data/         仅数据格式说明，不包含数据
docs/         来源、许可状态和发布检查说明
```

## 来源与许可

来源与修改范围见 [NOTICE](NOTICE.md)。当前无法确认上游可再分发许可证文本；**本目录是发布准备稿，许可证核实前不要作为已获授权的开源版本公开发布**。上游README的MIT徽章不能替代许可证文件或权利人的授权。没有替上游代码新增或冒认MIT许可。
