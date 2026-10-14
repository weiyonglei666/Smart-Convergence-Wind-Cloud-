from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..prompts import PLAN_AND_RETRIEVE_PROMPT, RETRIEVE_AND_PLAN_PROMPT
from .utils import extract_answer, format_chunks

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch
    from ..retrieval.retriever import Retriever


class RetrieveAndPlanGenerator:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = RETRIEVE_AND_PLAN_PROMPT | llm | StrOutputParser()

    def generate(self, query: str, chunks: List[Document]) -> str:
        return extract_answer(self.chain.invoke({"query": query, "chunks": format_chunks(chunks)}))


class PlanAndRetrieveGenerator:
    def __init__(self, llm: "ChatVllmBeamSearch", retriever: "Retriever"):
        self.llm = llm
        self.retriever = retriever
        self.chain = PLAN_AND_RETRIEVE_PROMPT | llm | StrOutputParser()

    def generate(self, query: str, top_k: Optional[int] = None) -> str:
        stage1 = self.chain.invoke({"query": query, "chunks": "(to be retrieved)"})
        hint = self._extract_hint(stage1, query)
        chunks = self.retriever.retrieve(hint, top_k=top_k)
        return extract_answer(self.chain.invoke({"query": query, "chunks": format_chunks(chunks)}))

    @staticmethod
    def _extract_hint(text: str, query: str) -> str:
        upper = text.upper()
        idx = upper.find("RETRIEVAL HINT")
        if idx < 0:
            return query
        tail = text[idx + len("RETRIEVAL HINT"):]
        end = tail.upper().find("ANSWER")
        block = tail[:end] if end >= 0 else tail
        hints = [ln.strip(" -*\t") for ln in block.splitlines() if ln.strip(" -*\t")]
        return " ".join(hints) if hints else query
