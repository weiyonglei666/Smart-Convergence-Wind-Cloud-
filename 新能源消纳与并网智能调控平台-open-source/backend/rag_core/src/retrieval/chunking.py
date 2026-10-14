from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

_TOKENIZERS: Dict[str, Any] = {}


def _tokenizer(path: str):
    if path not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[path] = AutoTokenizer.from_pretrained(path)
    return _TOKENIZERS[path]


class TokenChunkSplitter(TextSplitter):
    def __init__(self, tokenizer_path: str, chunk_size: int = 256, sliding_window: int = 0):
        super().__init__(chunk_size=chunk_size, chunk_overlap=sliding_window)
        self._hf_tokenizer = _tokenizer(tokenizer_path)

    def split_text(self, text: str) -> List[str]:
        ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
        step = self._chunk_size - self._chunk_overlap if self._chunk_overlap > 0 else self._chunk_size
        chunks: List[str] = []
        for start in range(0, len(ids), step):
            window = ids[start : start + self._chunk_size]
            if not window:
                break
            chunks.append(self._hf_tokenizer.decode(window).strip())
            if start + self._chunk_size >= len(ids):
                break
        return chunks

    def split_document(self, text: str, doc_id: str = "") -> List[Document]:
        return [
            Document(page_content=t, metadata={"chunk_id": i, "doc_id": doc_id})
            for i, t in enumerate(self.split_text(text))
        ]
