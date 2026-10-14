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
    graph: RhetoricalGraph | None
    plan: Plan | None


class CloudDiscoRAGPipeline:
    """Adapted Disco-RAG structure pipeline backed by an API chat model."""

    def __init__(self, index: VectorIndex, llm: ChatCloudLLM, top_k: int = 5, *, use_rst: bool = True, use_graph: bool = True, use_planning: bool = True):
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.use_rst = use_rst
        self.use_graph = use_graph
        self.use_planning = use_planning
        # Intermediate structure generation only needs compact machine-readable
        # outputs; keep the final answer budget on the original LLM.
        stage_llm = ChatCloudLLM(llm._client, llm.model_name, output_language=llm.output_language, max_tokens_override=256)
        self.rst_parser = RstParser(stage_llm)
        self.graph_builder = GraphBuilder(stage_llm)
        self.planner = Planner(stage_llm)
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
        trees = self.rst_parser.parse_chunks(chunks) if self.use_rst else []
        graph = self.graph_builder.build(chunks) if self.use_graph else None
        plan = self.planner.plan(query, chunks, trees, graph) if self.use_planning else None
        answer = self.generator.generate(query, chunks, trees, graph, plan)
        return DiscoPipelineResult(answer, [hit.document.source_id for hit in hits], trees, graph, plan)
