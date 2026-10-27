from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PowerEvalConfig:
    api_key: str
    model: str = "Qwen3.8-27B"
    base_url: str = "http://127.0.0.1:8002/v1"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 120.0
    top_k: int = 10
    corpus_root: Path = Path("data/corpus")
    benchmark_root: Path = Path("data/benchmark")

    @property
    def api_model(self) -> str:
        """Provider model identifier; keep the requested display name for reports."""
        aliases = {"DeepSeek-V4.1-Flash": "deepseek-flash", "deepseek-v4.1-flash": "deepseek-flash"}
        return aliases.get(self.model, self.model)

    @classmethod
    def from_env(cls, require_api_key: bool = True) -> "PowerEvalConfig":
        # Generic names take precedence; DeepSeek names remain supported for backward compatibility.
        key = os.getenv("POWER_LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")).strip()
        if require_api_key and not key:
            raise RuntimeError("POWER_LLM_API_KEY (or DEEPSEEK_API_KEY) is not set; put the key in the environment, never in source code")
        return cls(
            api_key=key,
            model=os.getenv("POWER_LLM_MODEL", "Qwen3.8-27B"),
            base_url=os.getenv("POWER_LLM_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:8001/v1")).rstrip("/"),
            temperature=float(os.getenv("POWER_LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("POWER_LLM_MAX_TOKENS", "1024")),
            timeout_seconds=float(os.getenv("POWER_LLM_TIMEOUT", "120")),
            top_k=int(os.getenv("POWER_RAG_TOP_K", "10")),
            corpus_root=Path(os.getenv("POWER_CORPUS_ROOT", "data/corpus")),
            benchmark_root=Path(os.getenv("ELECBENCH_ROOT", "data/benchmark")),
        )
