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
        return self.index.search(question, 10)

    def answer(self, question: str) -> AnswerResult:
        hits = self.retrieve(question)
        prompt = self.prompt(question, hits)
        answer = self.generator.generate([{"role": "system", "content": "你是严谨的电力系统问答助手。只依据给定资料回答。"}, {"role": "user", "content": prompt}])
        return AnswerResult(answer, self.name, [h.document.source_id for h in hits], [h.score for h in hits])

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"问题：{question}\n资料：\n{_context(hits)}\n请给出简洁、可核验的答案。"


class DirectAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex):
        super().__init__(generator, index, "direct")

    def retrieve(self, question: str) -> list[SearchHit]:
        return []

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"问题：{question}\n请直接回答，并在不确定时明确说明。"


class StandardRAGAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex):
        super().__init__(generator, index, "standard_rag")


class GraphRAGAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, graph: PowerKnowledgeGraph):
        super().__init__(generator, graph.index, "graphrag")
        self.graph = graph

    def retrieve(self, question: str) -> list[SearchHit]:
        return self.graph.retrieve(question, 10)


class PlannedAnswerer(BaseAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex, mode: str):
        super().__init__(generator, index, mode)

    def prompt(self, question: str, hits: list[SearchHit]) -> str:
        return f"先列出回答问题所需的关键事实，再逐项核对资料，最后给出结论。\n问题：{question}\n资料：\n{_context(hits)}"


class DiscoRAGAnswerer(PlannedAnswerer):
    def __init__(self, generator: TextGenerator, index: VectorIndex):
        super().__init__(generator, index, "disco_rag")
        model_name = getattr(getattr(generator, "config", None), "model", "DeepSeek-V4.1-Flash")
        self.pipeline = CloudDiscoRAGPipeline(index, ChatCloudLLM(generator, model_name), top_k=10)

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
        return f"构建篇章结构：识别定义、条件、因果和结论之间的关系；仅使用资料推理。\n问题：{question}\n资料：\n{_context(hits)}\n输出最终答案，并标注使用的来源编号。"
