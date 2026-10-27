import re
from typing import TYPE_CHECKING, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..prompts import RST_TREE_PROMPT
from ..prompts.relations import INTRA_RELATIONS
from .cache import MissingRstCache, RstCache
from .tree import EDU, RstRelation, RstTree

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch

EDU_RE = re.compile(r"\[(\d+)\]\s*(.+)")
REL_RE = re.compile(r"RELATION\(\s*EDU[_ ]?(\d+)\s*,\s*EDU[_ ]?(\d+)\s*\)\s*:\s*\{?([A-Z_]+)\}?")
ROLE_RE = re.compile(r"(NUCLEUS|SATELLITE)\[(\d+)\]")

# Backwards-compatible symbol; the durable cache lives in RstCache.
_OFFLINE_CACHE: Dict[Tuple[str, str], str] = {}


class RstParser:
    def __init__(self, llm: "ChatVllmBeamSearch", cache: RstCache | None = None, require_cache: bool = False):
        self.llm = llm
        self.chain = RST_TREE_PROMPT | llm | StrOutputParser()
        self.cache = cache or RstCache()
        self.require_cache = require_cache

    def parse_chunks(self, chunks: List[Document]) -> List[RstTree]:
        model = self.llm.model_name
        outputs: dict[str, str] = {}
        pending: list[Document] = []
        for chunk in chunks:
            cached = self.cache.get(model, chunk.page_content)
            if cached is None:
                if self.require_cache:
                    raise MissingRstCache("RST cache miss in an offline-only run")
                pending.append(chunk)
            else:
                outputs[chunk.page_content] = cached
        if pending:
            outs = self.chain.batch([{"chunk": chunk.page_content} for chunk in pending])
            for chunk, raw in zip(pending, outs):
                self.cache.put(model, chunk.page_content, raw)
                outputs[chunk.page_content] = raw
        return [self._parse_output(outputs[chunk.page_content], chunk.metadata["chunk_id"]) for chunk in chunks]

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
