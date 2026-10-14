import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..prompts import PLANNING_PROMPT
from ..rhetorical_graph.graph import RhetoricalGraph, format_chunks
from ..rst_tree.tree import RstTree

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch

SENT_RE = re.compile(r"(?<=[.!?])\s+")


def serialize_trees(trees: List[RstTree]) -> str:
    return "\n".join(t.serialize() for t in trees)


@dataclass
class Plan:
    text: str = ""
    steps: List[str] = field(default_factory=list)
    raw: str = ""

    def serialize(self) -> str:
        if self.steps:
            return "[Discourse-aware Plan]\n" + " ".join(self.steps)
        return "[Discourse-aware Plan]\n" + self.text


class Planner:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = PLANNING_PROMPT | llm | StrOutputParser()

    def plan(self, query: str, chunks: List[Document], trees: List[RstTree], graph: RhetoricalGraph) -> Plan:
        out = self.chain.invoke(
            {
                "query": query,
                "chunks": format_chunks(chunks),
                "rst_trees": serialize_trees(trees) if trees else "None",
                "graph": graph.serialize() if graph else "None",
            }
        )
        return self._parse(out)

    @staticmethod
    def _parse(text: str) -> Plan:
        body = text
        idx = text.upper().find("PLAN:")
        if idx >= 0:
            body = text[idx + len("PLAN:"):].strip()
        steps = [s.strip() for s in SENT_RE.split(body) if s.strip()]
        return Plan(text=body, steps=steps, raw=text)
