from typing import List

from langchain_core.documents import Document


def format_chunks(chunks: List[Document]) -> str:
    return "\n\n".join(f"[Chunk {i + 1}] {c.page_content}" for i, c in enumerate(chunks))


def extract_answer(text: str) -> str:
    upper = text.upper()
    for marker in ("ANSWER:", "ANSWER"):
        idx = upper.find(marker)
        if idx >= 0:
            return text[idx + len(marker):].strip()
    return text.strip()
