from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_eval.answerers import DirectAnswerer, DiscoRAGAnswerer, GraphRAGAnswerer, PlannedAnswerer, StandardRAGAnswerer
from power_eval.config import PowerEvalConfig
from power_eval.corpus import load_knowledge_corpus
from power_eval.elecbench import load_elecbench, run_resumable
from power_eval.graph import PowerKnowledgeGraph
from power_eval.index import FaissVectorIndex, VectorIndex
from power_eval.llm_client import DeepSeekClient
from src.rst_tree.cache import RstCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible power-domain RAG evaluation")
    method_names = ['disco_full', 'without_rst', 'without_graph', 'without_planning', 'standard_rag']
    parser.add_argument("--methods", nargs="+", choices=method_names, default=method_names)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/power_eval")
    parser.add_argument("--retriever", choices=["faiss", "tfidf"], default="faiss",
                        help="faiss is the documented retriever; tfidf is a debug-only fallback")
    parser.add_argument("--max-items", type=int, default=0, help="evaluate only the first N questions (0 = all)")
    parser.add_argument("--start-index", type=int, default=0, help="zero-based start index after loading questions")
    parser.add_argument("--ablation", action="store_true", help="Compatibility flag; five-method ablation is always enabled")
    args = parser.parse_args()
    args.ablation = True
    if args.start_index < 0 or args.max_items < 0:
        parser.error('sample offsets and counts must be nonnegative')
    cfg = PowerEvalConfig.from_env(require_api_key=not args.dry_run)
    faiss_dir = os.getenv("POWER_FAISS_INDEX", "")
    documents = [] if args.retriever == "faiss" else load_knowledge_corpus(cfg.corpus_root)
    items = load_elecbench(cfg.benchmark_root)
    if args.max_items > 0:
        items = items[args.start_index : args.start_index + args.max_items]
    elif args.start_index > 0:
        items = items[args.start_index :]
    print(f"Knowledge documents: {len(documents) if documents else 'FAISS index'}")
    print(f"Evaluation questions: {len(items)}")
    print(f"Model: {cfg.model}")
    print(f"Methods: {', '.join(args.methods)}")
    if args.dry_run:
        return
    if args.retriever == "faiss":
        if not faiss_dir:
            parser.error("POWER_FAISS_INDEX is required for the documented faiss retriever; "
                         "use --retriever tfidf only for debugging")
        index = FaissVectorIndex(
            faiss_dir,
            os.getenv("POWER_EMBEDDING_MODEL", "models/Qwen3-Embedding-8B"),
            batch_size=int(os.getenv("POWER_EMBEDDING_BATCH_SIZE", "2")),
        )
        print(f"Retriever: FAISS/Qwen3-Embedding-8B ({faiss_dir})")
    else:
        index = VectorIndex(documents)
        print("Retriever: legacy TF-IDF (debug-only; not the report configuration)")
    rst_cache_path = os.getenv("POWER_RST_CACHE", "").strip()
    require_rst_cache = os.getenv("POWER_REQUIRE_RST_CACHE", "0") == "1"
    if require_rst_cache and not rst_cache_path:
        parser.error("POWER_REQUIRE_RST_CACHE=1 requires POWER_RST_CACHE pointing to the offline cache")
    rst_cache = RstCache.load(rst_cache_path) if rst_cache_path else None
    if rst_cache is not None:
        print(f"RST cache: {rst_cache_path} ({len(rst_cache.entries)} entries)")
    graph = PowerKnowledgeGraph(documents, index) if documents else None
    client = DeepSeekClient(cfg)
    builders = {
        "direct": lambda: DirectAnswerer(client, index),
        "standard_rag": lambda: StandardRAGAnswerer(client, index),
        "graphrag": lambda: GraphRAGAnswerer(client, graph),
        "retrieve_plan": lambda: PlannedAnswerer(client, index, "retrieve_plan"),
        "plan_retrieve": lambda: PlannedAnswerer(client, index, "plan_retrieve"),
        "disco_rag": lambda: DiscoRAGAnswerer(client, index),
    }
    if args.ablation:
        from power_eval.ablation import METHODS, AblationAnswerer
        if "powergridqa_qwen" in args.output_dir and "ablation" not in args.output_dir:
            raise ValueError('Use a separate ablation output directory to avoid mixing old implementations')
        def build_ablation(name):
            return AblationAnswerer(DeepSeekClient(cfg), index, name, rst_cache=rst_cache,
                                    require_rst_cache=require_rst_cache)
        builders = {name: (lambda name=name: build_ablation(name)) for name in METHODS}
    for method in args.methods:
        if method not in builders:
            raise ValueError(f"Unknown method: {method}")
        run_resumable(items, builders[method]().answer, Path(args.output_dir) / f"{method}.jsonl")


if __name__ == "__main__":
    main()
