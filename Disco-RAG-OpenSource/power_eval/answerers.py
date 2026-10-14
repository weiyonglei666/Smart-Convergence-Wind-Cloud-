from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .corpus import PowerDocument
from .cloud_llm import ChatCloudLLM
from .disco_pipeline import CloudDiscoRAGPipeline
from .graph import PowerKnowledgeGraph
from .index import SearchHit, VectorIndex


class TextGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    method: str
    source_ids: list[str]
    retrieval_scores: list[float]
    metadata: dict[str, str] = field(default_factory=dict)


def _context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{hit.document.source_id}] {hit.document.text}" for hit in hits)


class BaseAnswerer:
    def __init__(self, generator: TextGenerator, index: VectorIndex, name: str):
        self.generator, self.index, self.name = generator, index, name

    def retrieve(self, question: str) -> list[SearchHit]:
        return self.index.search(question, 5)

    def answer(self, question: str) -> AnswerResult:
        hits = self.retrieve(question)
        prompt = self.prompt(question, hits)
        answer = self.generator.generate([{"role": "system", "content": "You are a rigorous power-systems question answering assistant. Answer in English. Prioritize the supplied material; if it does not cover the question, use reliable general knowledge or explicitly stated assumptions and label it as 'Not covered by the knowledge base; based on general knowledge/assumptions'. Do not present unsupported claims as conclusions from the material; state uncertainty when needed."}, {"role": "user", "content": prompt}])
        return AnswerResult(answer, self.name, [h.document.source_id for h in hits], [h.score for h in hits])

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"Question: {question}\nMaterials:\n{_context(hits)}\nPrioritize the materials; if insufficient, add reliable general knowledge and clearly label that the knowledge base does not cover it. Answer in concise, verifiable English."


class DirectAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex):
        super().__init__(generator, index, "direct")

    def retrieve(self, question: str) -> list[SearchHit]:
        return []

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"Question: {question}\nAnswer directly. When the knowledge base has no coverage, use reliable general knowledge and begin with 'Not covered by the knowledge base; based on general knowledge'. Answer in English and list assumptions for calculations."


class StandardRAGAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex):
        super().__init__(generator, index, "standard_rag")


class GraphRAGAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, graph: PowerKnowledgeGraph):
        super().__init__(generator, graph.index, "graphrag")
        self.graph = graph

    def retrieve(self, question: str) -> list[SearchHit]:
        return self.graph.retrieve(question, 5)


class PlannedAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex, mode: str):
        super().__init__(generator, index, mode)

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"First list the key facts needed to answer the question, then verify them against the materials, and finally give the conclusion. Answer in English.\nQuestion: {question}\nMaterials:\n{_context(hits)}"


class DiscoRAGAnswerer(PlannedAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex):
        super().__init__(generator, index, "disco_rag")
        model_name = getattr(getattr(generator, "config", None), "model", "DeepSeek-V4.1-Flash")
        self.pipeline = CloudDiscoRAGPipeline(index, ChatCloudLLM(generator, model_name), top_k=5)

    def answer(self, question: str) -> AnswerResult:
        result = self.pipeline.run(question)
        return AnswerResult(
            result.answer,
            self.name,
            result.source_ids,
            [0.0] * len(result.source_ids),
            {"rst_trees": "\n".join(tree.serialize() for tree in result.trees), "rhetorical_graph": result.graph.serialize(), "discourse_plan": result.plan.serialize()},
        )

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"Build a discourse structure by identifying definitions, conditions, causes, and conclusions; reason primarily from the materials. If insufficient, add reliable general knowledge and label that the knowledge base does not cover it. Answer in English.\nQuestion: {question}\nMaterials:\n{_context(hits)}\nOutput the final answer and cite the source IDs used."
