from typing import TYPE_CHECKING, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..prompts import STANDARD_RAG_PROMPT
from .utils import extract_answer, format_chunks

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch


class StandardRagGenerator:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = STANDARD_RAG_PROMPT | llm | StrOutputParser()

    def generate(self, query: str, chunks: List[Document]) -> str:
        return extract_answer(self.chain.invoke({"query": query, "chunks": format_chunks(chunks)}))
