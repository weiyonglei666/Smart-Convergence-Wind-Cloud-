from __future__ import annotations

import re
from collections import defaultdict

from .corpus import PowerDocument
from .index import SearchHit, VectorIndex

ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}")


class PowerKnowledgeGraph:
    def __init__(self, documents: list[PowerDocument], index: VectorIndex | None = None):
        self.documents = documents
        self.index = index or VectorIndex(documents)
        self.entity_to_docs: dict[str, set[str]] = defaultdict(set)
        self.doc_by_id = {doc.source_id: doc for doc in documents}
        for document in documents:
            for entity in self.extract_entities(document.text):
                self.entity_to_docs[entity].add(document.source_id)

    @staticmethod
    def extract_entities(text: str) -> set[str]:
        return {match.lower() for match in ENTITY_RE.findall(text)}

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_entities = self.extract_entities(query)
        graph_ids = {source_id for entity in query_entities for source_id in self.entity_to_docs.get(entity, set())}
        graph_hits = [SearchHit(self.doc_by_id[source_id], 1.0) for source_id in sorted(graph_ids)]
        vector_hits = self.index.search(query, top_k=max(top_k * 2, 10))
        seen = {hit.document.source_id for hit in graph_hits}
        merged = graph_hits + [hit for hit in vector_hits if hit.document.source_id not in seen]
        return merged[: max(1, top_k)]
