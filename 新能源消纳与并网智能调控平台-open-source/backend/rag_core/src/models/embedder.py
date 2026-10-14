from typing import Any, Dict, List

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

QUERY_INSTRUCTION = "Instruct: Given a query, retrieve relevant passages.\nQuery: "


class QwenEmbeddings(Embeddings):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.max_length = cfg.get("max_length", 8192)
        self.normalize = cfg.get("normalize", True)
        self.pooling = cfg.get("pooling", "last_token")
        self.batch_size = cfg.get("batch_size", 32)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["path"], padding_side="left")
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        self.model = AutoModel.from_pretrained(cfg["path"], torch_dtype=dtype, device_map="auto")
        self.model.eval()

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last_token":
            left_padded = mask[:, -1].sum() == mask.size(0)
            if left_padded:
                return hidden[:, -1]
            lengths = mask.sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        masked = hidden * mask.unsqueeze(-1)
        return masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> List[List[float]]:
        vecs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)
            out = self.model(**enc).last_hidden_state
            pooled = self._pool(out, enc["attention_mask"])
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.extend(pooled.float().cpu().tolist())
        return vecs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([QUERY_INSTRUCTION + text])[0]
