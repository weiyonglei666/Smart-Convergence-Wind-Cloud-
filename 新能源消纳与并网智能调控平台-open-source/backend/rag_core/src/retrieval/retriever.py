from typing import TYPE_CHECKING, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from ..config import RetrievalConfig
from .chunking import TokenChunkSplitter

if TYPE_CHECKING:
    from ..models.embedder import QwenEmbeddings


class Retriever:
    def __init__(self, embeddings: "QwenEmbeddings", cfg: RetrievalConfig, tokenizer_path: str):
        self.embeddings = embeddings
        self.cfg = cfg
        self.splitter = TokenChunkSplitter(
            tokenizer_path,
            chunk_size=cfg.chunk_size,
            sliding_window=cfg.sliding_window,
        )
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []

    def _distance_strategy(self) -> DistanceStrategy:
        if self.cfg.similarity == "cosine":
            return DistanceStrategy.MAX_INNER_PRODUCT
        return DistanceStrategy.EUCLIDEAN_DISTANCE

    def build_from_document(self, document: str, doc_id: str = "") -> "Retriever":
        return self.build_from_chunks(self.splitter.split_document(document, doc_id))

    def build_from_chunks(self, documents: List[Document]) -> "Retriever":
        self.documents = list(documents)
        self.vectorstore = FAISS.from_documents(
            self.documents,
            self.embeddings,
            distance_strategy=self._distance_strategy(),
        )
        return self

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        k = top_k or self.cfg.top_k
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def pool_chunks(self) -> List[Document]:
        return list(self.documents)
