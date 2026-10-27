"""Build the persistent intra-chunk RST cache used by online Disco-RAG runs.

The report states that RST trees for stable knowledge chunks are parsed at
knowledge-base build time.  Run this script once after the corpus and
Qwen3-Embedding tokenizer are available; online evaluation then reads the
cache instead of calling the model for every retrieved chunk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from power_eval.cloud_llm import ChatCloudLLM
from power_eval.config import PowerEvalConfig
from power_eval.llm_client import DeepSeekClient
from src.rst_tree.cache import RstCache
from src.rst_tree.parser import RstParser
from src.retrieval.chunking import TokenChunkSplitter


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and cache RST trees for the knowledge corpus")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--model", default="models/Qwen3-Embedding-8B", help="tokenizer path (same as the index build)")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", default="indexes/power_public_qwen3e8b_faiss/rst_cache.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = PowerEvalConfig.from_env(require_api_key=True)
    client = DeepSeekClient(cfg)
    stage_llm = ChatCloudLLM(client, cfg.model, output_language="English", max_tokens_override=256)
    splitter = TokenChunkSplitter(args.model, chunk_size=args.chunk_size, sliding_window=0)
    cache = RstCache.load(args.output) if Path(args.output).exists() else RstCache()
    parser_obj = RstParser(stage_llm, cache=cache, require_cache=False)

    chunks: list[Document] = []
    for suffix in ("*.txt", "*.md"):
        for source in sorted(Path(args.corpus).rglob(suffix)):
            text = source.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            for chunk_id, chunk in enumerate(splitter.split_text(text)):
                if chunk.strip():
                    chunks.append(Document(page_content=chunk, metadata={"chunk_id": chunk_id, "source_id": source.stem}))
    if args.limit:
        chunks = chunks[:args.limit]
    if not chunks:
        raise SystemExit("No knowledge chunks found; check --corpus and the tokenizer path")
    before = len(cache.entries)
    parser_obj.parse_chunks(chunks)
    path = cache.save(args.output)
    manifest = {
        "format": "rst-cache-v1",
        "output": str(path),
        "chunks": len(chunks),
        "new_entries": len(cache.entries) - before,
        "total_entries": len(cache.entries),
        "model": cfg.model,
        "chunk_size": args.chunk_size,
    }
    Path(args.output).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
