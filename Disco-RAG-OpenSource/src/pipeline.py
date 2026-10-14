from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .config import RetrievalConfig, RunConfig
from .generation.disco_rag import DiscoRagGenerator
from .perturb import inject_noise, perturb_graph, perturb_plan, perturb_rst
from .planning.planner import Plan, Planner
from .retrieval.retriever import Retriever
from .rhetorical_graph.graph import GraphBuilder, RhetoricalGraph
from .rst_tree.parser import RstParser
from .rst_tree.tree import RstTree

if TYPE_CHECKING:
    from .models.embedder import QwenEmbeddings
    from .models.llm import ChatVllmBeamSearch


@dataclass
class PipelineModules:
    use_rst: bool = True
    use_graph: bool = True
    use_plan: bool = True


class DiscoRagPipeline:
    def __init__(
        self,
        run_cfg: RunConfig,
        embedder: "QwenEmbeddings",
        structurer: "ChatVllmBeamSearch",
        generator: Optional["ChatVllmBeamSearch"] = None,
        modules: Optional[PipelineModules] = None,
        chunk_size: Optional[int] = None,
        graph_builder: Optional[GraphBuilder] = None,
    ):
        self.cfg = run_cfg
        self.embedder = embedder
        self.structurer = structurer
        self.generator = generator or structurer
        self.modules = modules or PipelineModules()
        retrieval_cfg = RetrievalConfig(
            chunk_size=chunk_size or run_cfg.retrieval.chunk_size,
            sliding_window=run_cfg.retrieval.sliding_window,
            top_k=run_cfg.retrieval.top_k,
            similarity=run_cfg.retrieval.similarity,
        )
        tok_path = self.generator.model_path
        self.retriever = Retriever(embedder, retrieval_cfg, tok_path)
        self.parser = RstParser(self.structurer)
        self.graph_builder = graph_builder or GraphBuilder(self.structurer)
        self.planner = Planner(self.structurer)
        self.gen = DiscoRagGenerator(self.generator)

    def index_document(self, document: str, doc_id: str = "") -> "DiscoRagPipeline":
        self.retriever.build_from_document(document, doc_id)
        return self

    def build_structures(self, query: str, top_k: Optional[int] = None, noise: float = 0.0, seed: int = 42):
        chunks = self.retriever.retrieve(query, top_k=top_k)
        if noise > 0:
            chunks = inject_noise(chunks, self.retriever.pool_chunks(), noise, seed)
        trees = self.parser.parse_chunks(chunks) if self.modules.use_rst else None
        graph = self.graph_builder.build(chunks) if self.modules.use_graph else None
        plan = self.planner.plan(query, chunks, trees or [], graph) if self.modules.use_plan else None
        return chunks, trees, graph, plan

    def apply_perturbation(self, trees, graph, plan, spec, seed: int = 42):
        if not spec:
            return trees, graph, plan
        target, mode, ratio = spec["target"], spec["mode"], spec.get("ratio", 0.3)
        if target == "rst" and trees is not None:
            trees = perturb_rst(trees, mode, ratio, seed)
        elif target == "graph" and graph is not None:
            graph = perturb_graph(graph, mode, ratio, seed)
        elif target == "plan" and plan is not None:
            plan = perturb_plan(plan, mode, ratio, seed)
        return trees, graph, plan

    def run(self, query: str, top_k: Optional[int] = None, noise: float = 0.0, perturb=None, seed: int = 42) -> str:
        chunks, trees, graph, plan = self.build_structures(query, top_k=top_k, noise=noise, seed=seed)
        trees, graph, plan = self.apply_perturbation(trees, graph, plan, perturb, seed)
        return self.gen.generate(query, chunks, trees, graph, plan)
