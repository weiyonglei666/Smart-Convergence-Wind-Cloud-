# DeepSeek + Disco-RAG Core

这是独立的电力 RAG 核心包，不会自动修改或注册数字化大屏后端接口。

## 内容

- `power_eval/`：DeepSeek 客户端、PowerGridQA 加载、检索、Disco-RAG 管线
- `src/`：官方 Disco-RAG 的 RST 树、跨块修辞图、规划器和生成器
- `../data/powergridqa/`：PowerGridQA 电力知识库

## 配置

```powershell
$env:DEEPSEEK_API_KEY="你的Key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:POWER_LLM_MODEL="DeepSeek-V4.1-Flash"
```

代码会自动把展示名映射为 DeepSeek API 的请求模型标识。

## 接入方式

在你自己的服务中将 `rag_core` 加入 Python 模块路径，然后使用：

```python
from power_eval.cloud_llm import ChatCloudLLM
from power_eval.config import PowerEvalConfig
from power_eval.corpus import load_powergridqa
from power_eval.disco_pipeline import CloudDiscoRAGPipeline
from power_eval.index import VectorIndex
from power_eval.llm_client import DeepSeekClient

cfg = PowerEvalConfig.from_env()
documents = load_powergridqa("backend/data/powergridqa")
index = VectorIndex(documents)
llm = ChatCloudLLM(DeepSeekClient(cfg), cfg.model)
pipeline = CloudDiscoRAGPipeline(index, llm, top_k=cfg.top_k)
result = pipeline.run("你的电力问题")
print(result.answer)
```

API Key 只从环境变量读取，不写入本目录。
