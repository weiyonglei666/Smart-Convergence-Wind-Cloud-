from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser

from ..prompts import FULL_CONTEXT_PROMPT
from .utils import extract_answer

if TYPE_CHECKING:
    from ..models.llm import ChatVllmBeamSearch

PROMPT_MARGIN = 1024


class FullContextGenerator:
    def __init__(self, llm: "ChatVllmBeamSearch"):
        self.llm = llm
        self.chain = FULL_CONTEXT_PROMPT | llm | StrOutputParser()

    def generate(self, query: str, document: str) -> str:
        budget = self.llm.max_model_len - self.llm.max_new_tokens - PROMPT_MARGIN
        document = self.llm.truncate(document, budget)
        return extract_answer(self.chain.invoke({"query": query, "document": document}))
