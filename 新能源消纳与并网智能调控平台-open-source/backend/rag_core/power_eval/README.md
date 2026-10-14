# Power RAG Evaluation

本实验将 PowerGridQA 作为电力知识库，将 ElecBench 作为独立测试集，对比 Direct LLM、Standard RAG、GraphRAG、Retrieve-and-Plan、Plan-and-Retrieve 和 Disco-RAG。

Disco-RAG 方法调用仓库中的官方组件：`RstParser` 生成每个检索块的 RST 树，`GraphBuilder` 生成跨块修辞图，`Planner` 生成 discourse-aware plan，`DiscoRagGenerator` 根据三类结构生成最终答案。云端模型通过 `ChatCloudLLM` 适配器接入，不依赖本地 vLLM。

在 `Disco-RAG` 目录执行：

```powershell
$env:DEEPSEEK_API_KEY="你的新Key"
python scripts/run_power_eval.py --dry-run --methods direct standard_rag graphrag retrieve_plan plan_retrieve disco_rag
python scripts/run_power_eval.py --methods direct standard_rag graphrag retrieve_plan plan_retrieve disco_rag
```

密钥只从环境变量读取，不写入代码、配置或结果。评测结果按方法写入 `results/power_eval/*.jsonl`，支持中断后继续运行；Disco-RAG 结果额外保存 RST 树、修辞图和 discourse plan，便于审计和消融实验。

## 切换到本地 OpenAI 兼容服务

不需要修改代码，在当前 PowerShell 会话中设置：

```powershell
$env:POWER_LLM_API_KEY="本地服务的Key"
$env:POWER_LLM_BASE_URL="http://192.168.1.130:8001/v1"
$env:POWER_LLM_MODEL="Qwen3.8-27B"
```

未设置 `POWER_LLM_*` 时，程序继续使用原有的 DeepSeek 配置。两套配置互不覆盖实验结果。
