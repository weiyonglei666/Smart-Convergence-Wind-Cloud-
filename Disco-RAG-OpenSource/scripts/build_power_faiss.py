"""Build the public power-domain FAISS index used by the evaluation runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.models.embedder import QwenEmbeddings
from src.retrieval.chunking import TokenChunkSplitter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--model", default="models/Qwen3-Embedding-8B")
    ap.add_argument("--output", default="indexes/power_public_qwen3e8b_faiss")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    texts = sorted(corpus.rglob("*.txt"))
    if not texts:
        raise SystemExit(f"No .txt source files found under {corpus}")

    splitter = TokenChunkSplitter(str(model), chunk_size=args.chunk_size, sliding_window=0)
    chunks: list[Document] = []
    for source in texts:
        text = source.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        for local_id, chunk in enumerate(splitter.split_text(text)):
            if chunk.strip():
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source_file": str(source.relative_to(corpus)),
                        "source_id": source.stem,
                        "chunk_id": local_id,
                        "chunk_tokens": args.chunk_size,
                    },
                ))
    if not chunks:
        raise SystemExit("Chunking produced no documents")

    embedder = QwenEmbeddings({
        "path": str(model),
        "dtype": "bfloat16",
        "max_length": 8192,
        "pooling": "last_token",
        "normalize": True,
        "batch_size": args.batch_size,
    })
    store = FAISS.from_documents(
        chunks,
        embedder,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    output.mkdir(parents=True, exist_ok=True)
    store.save_local(str(output))
    evidence = {f"{d.metadata['source_file']}:{d.metadata['chunk_id']}": d.page_content for d in chunks}
    (output / "evidence_docs.json").write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "embedding_model": str(model),
        "chunk_size": args.chunk_size,
        "similarity": "cosine_via_normalized_inner_product",
        "top_k": 10,
        "source_files": len(texts),
        "chunks": len(chunks),
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
