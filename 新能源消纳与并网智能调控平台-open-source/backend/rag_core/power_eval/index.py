from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

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
        # Keep Latin/number terms intact (AGC, NWP, FD0001, 15min) while
        # splitting Chinese into overlapping unigrams/bigrams.  The original
        # contiguous-Unicode tokenization made a query such as “限电消纳” fail
        # to match documents containing “限电” and “消纳” separately.
        tokens: list[str] = []
        for chunk in TOKEN_RE.findall(text):
            chunk = chunk.lower()
            if any("\u4e00" <= char <= "\u9fff" for char in chunk):
                tokens.extend(char for char in chunk if "\u4e00" <= char <= "\u9fff")
                tokens.extend(chunk[index:index + 2] for index in range(len(chunk) - 1) if all("\u4e00" <= char <= "\u9fff" for char in chunk[index:index + 2]))
            else:
                tokens.append(chunk)
        return tokens

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
