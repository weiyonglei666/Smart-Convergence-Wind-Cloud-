import re
from typing import TYPE_CHECKING, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..prompts import RST_TREE_PROMPT
from ..prompts.relations import INTRA_RELATIONS
from .tree import EDU, RstRelation, RstTree

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch

EDU_RE = re.compile(r"\[(\d+)\]\s*(.+)")
REL_RE = re.compile(r"RELATION\(\s*EDU[_ ]?(\d+)\s*,\s*EDU[_ ]?(\d+)\s*\)\s*:\s*\{?([A-Z_]+)\}?")
ROLE_RE = re.compile(r"(NUCLEUS|SATELLITE)\[(\d+)\]")

_OFFLINE_CACHE: Dict[Tuple[str, str], str] = {}


class RstParser:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = RST_TREE_PROMPT | llm | StrOutputParser()

    def parse_chunks(self, chunks: List[Document]) -> List[RstTree]:
        model = self.llm.model_name
        pending = [c for c in chunks if (model, c.page_content) not in _OFFLINE_CACHE]
        if pending:
            outs = self.chain.batch([{"chunk": c.page_content} for c in pending])
            for c, o in zip(pending, outs):
                _OFFLINE_CACHE[(model, c.page_content)] = o
        return [
            self._parse_output(_OFFLINE_CACHE[(model, c.page_content)], c.metadata["chunk_id"])
            for c in chunks
        ]

    @staticmethod
    def _parse_output(text: str, chunk_id: int) -> RstTree:
        edus: List[EDU] = []
        relations: List[RstRelation] = []
        section = None
        roles = {}
        for line in text.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("EDUS"):
                section = "edus"
                continue
            if upper.startswith("RST ANALYSIS"):
                section = "analysis"
                continue
            if upper.startswith("TREE STRUCTURE"):
                section = "tree"
                continue
            if section == "edus":
                m = EDU_RE.match(stripped)
                if m:
                    edus.append(EDU(index=int(m.group(1)), text=m.group(2).strip()))
            elif section == "analysis":
                m = REL_RE.search(stripped)
                if m and m.group(3) in INTRA_RELATIONS:
                    relations.append(RstRelation(int(m.group(1)), int(m.group(2)), m.group(3)))
            elif section == "tree":
                rm = ROLE_RE.search(stripped)
                if rm:
                    roles[int(rm.group(2))] = "N" if rm.group(1) == "NUCLEUS" else "S"
        for e in edus:
            if e.index in roles:
                e.role = roles[e.index]
        return RstTree(chunk_id=chunk_id, edus=edus, relations=relations, raw=text)
