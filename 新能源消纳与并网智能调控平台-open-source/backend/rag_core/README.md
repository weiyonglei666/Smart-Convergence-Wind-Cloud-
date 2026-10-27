# Platform RAG Core

平台后端内置的电力 RAG 核心包，由 `app/assistant/service.py` 调用。

## 内容

- `power_eval/`：OpenAI 兼容客户端、知识语料加载、TF-IDF 检索、Disco-RAG 管线
- `src/`：RST 树、跨块修辞图、规划器和生成器
- `../data/knowledge/`：平台知识语料（`.txt` / `.md`）
- `../data/powergridqa/`：评测集，仅用于评测，不参与检索

## 配置

```powershell
$env:POWER_LLM_API_KEY="你的Key"
$env:POWER_LLM_BASE_URL="http://127.0.0.1:8002/v1"
$env:POWER_LLM_MODEL="Qwen3.8-27B"
$env:POWER_RAG_TOP_K="10"
$env:RAG_MODE="disco"
$env:RST_CACHE="data/knowledge/.rst_cache.json"
```

生成服务使用独立端口 `8002`，不要占用平台 API 的 `8001`。

## 接入方式

```python
from rag_core.power_eval.cloud_llm import ChatCloudLLM
from rag_core.power_eval.config import PowerEvalConfig
from rag_core.power_eval.corpus import load_knowledge_corpus
from rag_core.power_eval.disco_pipeline import CloudDiscoRAGPipeline
from rag_core.power_eval.index import VectorIndex
from rag_core.power_eval.llm_client import DeepSeekClient
from rag_core.src.rst_tree.cache import RstCache

cfg = PowerEvalConfig.from_env()
documents = load_knowledge_corpus("data/knowledge")
index = VectorIndex(documents)
llm = ChatCloudLLM(DeepSeekClient(cfg), cfg.model)
pipeline = CloudDiscoRAGPipeline(index, llm, top_k=cfg.top_k, rst_cache=RstCache.load(cfg.rst_cache))
result = pipeline.run("你的电力问题")
print(result.answer)
```

API Key 只从环境变量读取，不写入本目录。
