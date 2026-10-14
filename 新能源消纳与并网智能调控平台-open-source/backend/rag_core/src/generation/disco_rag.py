from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..planning.planner import Plan, serialize_trees
from ..prompts import DISCO_RAG_PROMPT
from ..rhetorical_graph.graph import RhetoricalGraph
from ..rst_tree.tree import RstTree
from .utils import extract_answer, format_chunks

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch


class DiscoRagGenerator:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = DISCO_RAG_PROMPT | llm | StrOutputParser()

    def generate(
        self,
        query: str,
        chunks: List[Document],
        trees: Optional[List[RstTree]] = None,
        graph: Optional[RhetoricalGraph] = None,
        plan: Optional[Plan] = None,
    ) -> str:
        out = self.chain.invoke(
            {
                "query": query,
                "chunks": format_chunks(chunks),
                "rst_trees": serialize_trees(trees) if trees else "None",
                "graph": graph.serialize() if graph else "None",
                "plan": plan.serialize() if plan else "None",
            }
        )
        return extract_answer(out)
