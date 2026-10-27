# Power RAG Core (platform)

平台后端内置的 RAG 核心，供 `app/assistant/service.py` 使用。它与根目录的独立 `Disco-RAG-OpenSource` 实验目录职责不同：

- 知识库：`backend/data/knowledge/` 下的 `.txt` / `.md`；`backend/data/powergridqa/` 只放评测集，不参与检索。
- 检索：本地 TF-IDF 向量索引；正式离线实验仍使用独立目录中的 Qwen3-Embedding-8B / FAISS。
- 结构：`CloudDiscoRAGPipeline` 调用 `RstParser`、`GraphBuilder`、`Planner`，把块内 RST、跨块修辞图和话语规划写入最终报告提示词；`RAG_MODE=light` 可退回纯片段检索。
- 缓存：`RST_CACHE` 指向 `RstCache` 文件；存在时优先读取，缺失时在线解析并写回（正式评测应预构建并设置 `POWER_REQUIRE_RST_CACHE=1`）。

## 配置

```powershell
$env:POWER_LLM_API_KEY="本地服务的Key"
$env:POWER_LLM_BASE_URL="http://127.0.0.1:8002/v1"
$env:POWER_LLM_MODEL="Qwen3.8-27B"
$env:POWER_RAG_TOP_K="10"
$env:RAG_MODE="disco"
$env:RST_CACHE="data/knowledge/.rst_cache.json"
```

生成服务不要占用平台 FastAPI 的 `8001` 端口。密钥只从环境变量读取，不写入代码或配置。
