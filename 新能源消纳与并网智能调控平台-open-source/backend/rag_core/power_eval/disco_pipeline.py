from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.generation.disco_rag import DiscoRagGenerator
from src.planning.planner import Plan, Planner
from src.rhetorical_graph.graph import GraphBuilder, RhetoricalGraph
from src.rst_tree.parser import RstParser
from src.rst_tree.tree import RstTree

from .cloud_llm import ChatCloudLLM
from .index import VectorIndex


@dataclass(frozen=True)
class DiscoPipelineResult:
    answer: str
    source_ids: list[str]
    trees: list[RstTree]
    graph: RhetoricalGraph
    plan: Plan


class CloudDiscoRAGPipeline:
    """Official Disco-RAG structure pipeline backed by a cloud chat model."""

    def __init__(self, index: VectorIndex, llm: ChatCloudLLM, top_k: int = 5):
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.rst_parser = RstParser(llm)
        self.graph_builder = GraphBuilder(llm)
        self.planner = Planner(llm)
        self.generator = DiscoRagGenerator(llm)

    def run(self, query: str) -> DiscoPipelineResult:
        hits = self.index.search(query, self.top_k)
        chunks = [
            Document(
                page_content=hit.document.text,
                metadata={"chunk_id": index + 1, "source_id": hit.document.source_id, "score": hit.score},
            )
            for index, hit in enumerate(hits)
        ]
        trees = self.rst_parser.parse_chunks(chunks)
        graph = self.graph_builder.build(chunks)
        plan = self.planner.plan(query, chunks, trees, graph)
        answer = self.generator.generate(query, chunks, trees, graph, plan)
        return DiscoPipelineResult(answer, [hit.document.source_id for hit in hits], trees, graph, plan)
