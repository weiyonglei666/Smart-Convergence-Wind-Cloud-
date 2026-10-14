import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> Dict[str, Any]:
    if not name.endswith(".yaml"):
        name = name + ".yaml"
    path = name if os.path.isabs(name) else os.path.join(CONFIG_DIR, name)
    return load_yaml(path)


@dataclass
class ModelConfig:
    name: str
    path: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 131072
    gpu_memory_utilization: float = 0.9


@dataclass
class RetrievalConfig:
    chunk_size: int = 256
    sliding_window: int = 0
    top_k: int = 10
    similarity: str = "cosine"


@dataclass
class DecodingConfig:
    beam_width: int = 3
    max_new_tokens: int = 1024


@dataclass
class RunConfig:
    models: Dict[str, ModelConfig]
    embedder: Dict[str, Any]
    retrieval: RetrievalConfig
    decoding: DecodingConfig
    backbone: str = "llama3_70b"
    seed: int = 42

    @classmethod
    def from_default(cls) -> "RunConfig":
        cfg = load_config("default")
        models = {k: ModelConfig(**v) for k, v in cfg["models"].items()}
        return cls(
            models=models,
            embedder=cfg["embedder"],
            retrieval=RetrievalConfig(**cfg["retrieval"]),
            decoding=DecodingConfig(**cfg["decoding"]),
            backbone=cfg.get("backbone", "llama3_70b"),
            seed=cfg.get("seed", 42),
        )

    def model(self, key: str) -> ModelConfig:
        return self.models[key]
