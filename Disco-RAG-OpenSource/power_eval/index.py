from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .corpus import PowerDocument

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class SearchHit:
    document: PowerDocument
    score: float


class VectorIndex:
    """Deterministic TF-IDF index; replaceable by FAISS embeddings in production."""

    def __init__(self, documents: list[PowerDocument]):
        if not documents:
            raise ValueError("VectorIndex requires at least one document")
        self.documents = documents
        self._tokens = [self._tokenize(d.text) for d in documents]
        df = Counter(token for row in self._tokens for token in set(row))
        self._idf = {token: math.log((1 + len(documents)) / (1 + count)) + 1 for token, count in df.items()}
        self._vectors = [self._vector(row) for row in self._tokens]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vector = {token: (count / max(1, len(tokens))) * self._idf.get(token, 1.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_vector = self._vector(self._tokenize(query))
        scored = []
        for document, vector in zip(self.documents, self._vectors):
            score = sum(query_vector.get(token, 0.0) * value for token, value in vector.items())
            scored.append(SearchHit(document, score))
        return sorted(scored, key=lambda hit: (-hit.score, hit.document.source_id))[: max(1, top_k)]


class FaissVectorIndex:
    """FAISS index backed by the official Qwen3-Embedding-8B retriever."""

    def __init__(self, index_dir: str | Path, model_dir: str | Path, batch_size: int = 2):
        from langchain_community.vectorstores import FAISS
        from src.models.embedder import QwenEmbeddings

        self.index_dir = Path(index_dir)
        self.model_dir = Path(model_dir)
        self.embeddings = QwenEmbeddings({
            "path": str(self.model_dir),
            "dtype": "bfloat16",
            "max_length": 8192,
            "pooling": "last_token",
            "normalize": True,
            "batch_size": batch_size,
        })
        self.store = FAISS.load_local(
            str(self.index_dir), self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        rows = self.store.similarity_search_with_score(query, k=max(1, top_k))
        hits: list[SearchHit] = []
        for document, score in rows:
            md = document.metadata
            source_id = f"{md.get('source_file', md.get('source_id', 'unknown'))}:{md.get('chunk_id', 0)}"
            power_doc = PowerDocument(
                text=document.page_content,
                source_file=str(md.get('source_file', '')),
                source_id=source_id,
                metadata=dict(md),
            )
            hits.append(SearchHit(power_doc, float(score)))
        return hits
