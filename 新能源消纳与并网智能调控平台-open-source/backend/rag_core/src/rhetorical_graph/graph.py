import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..prompts import GRAPH_PROMPT
from ..prompts.relations import INTER_RELATIONS

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch

EDGE_RE = re.compile(r"CHUNK\[(\d+)\]\s*->\s*CHUNK\[(\d+)\]\s*:\s*\{?([A-Z_]+)\}?")


def format_chunks(chunks: List[Document]) -> str:
    return "\n".join(f"CHUNK[{i + 1}]: {c.page_content}" for i, c in enumerate(chunks))


@dataclass
class RhetoricalEdge:
    source: int
    target: int
    relation: str


@dataclass
class RhetoricalGraph:
    edges: List[RhetoricalEdge] = field(default_factory=list)
    num_chunks: int = 0
    raw: str = ""

    def related_edges(self) -> List[RhetoricalEdge]:
        return [e for e in self.edges if e.relation != "UNRELATED"]

    def serialize(self) -> str:
        lines = [f"CHUNK[{e.source}] -> CHUNK[{e.target}]: {e.relation}" for e in self.related_edges()]
        return "[Inter-chunk Rhetorical Graph]\n" + "\n".join(lines)


class GraphBuilder:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = GRAPH_PROMPT | llm | StrOutputParser()

    def build(self, chunks: List[Document]) -> RhetoricalGraph:
        out = self.chain.invoke({"chunks": format_chunks(chunks)})
        return self._parse_output(out, len(chunks))

    @staticmethod
    def _parse_output(text: str, num_chunks: int) -> RhetoricalGraph:
        edges: List[RhetoricalEdge] = []
        for line in text.splitlines():
            m = EDGE_RE.search(line.strip())
            if m and m.group(3) in INTER_RELATIONS:
                i, j = int(m.group(1)), int(m.group(2))
                if i != j:
                    edges.append(RhetoricalEdge(i, j, m.group(3)))
        return RhetoricalGraph(edges=edges, num_chunks=num_chunks, raw=text)
