import json

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from power_eval.corpus import load_knowledge_corpus
from src.rst_tree.cache import MissingRstCache, RstCache
from src.rst_tree.parser import RstParser

RST_OUTPUT = (
    "RST ANALYSIS\nRELATION(EDU1, EDU2): ELABORATION\n"
    "EDUS\n[1] 核心结论\n[2] 解释\nTREE STRUCTURE\nNUCLEUS[1]\nSATELLITE[2]\n"
)


class FakeLLM(Runnable):
    model_name = "fake-model"

    def __init__(self):
        self.calls = 0

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        return RST_OUTPUT

    def batch(self, inputs, config=None, **kwargs):
        self.calls += len(inputs)
        return [RST_OUTPUT for _ in inputs]


def test_knowledge_loader_reads_text_and_ignores_benchmark_jsonl(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "textbook.txt").write_text("电力系统稳定控制", encoding="utf-8")
    (corpus / "benchmark.jsonl").write_text(
        json.dumps({"question": "Q", "answer": "A"}), encoding="utf-8"
    )
    docs = load_knowledge_corpus(corpus)
    assert [doc.source_id for doc in docs] == ["textbook.txt"]
    assert "answer" not in docs[0].text


def test_rst_cache_roundtrip_and_missing_policy(tmp_path):
    cache_path = tmp_path / "rst_cache.json"
    cache = RstCache()
    cache.put("fake-model", "chunk text", RST_OUTPUT)
    cache.save(cache_path)
    loaded = RstCache.load(cache_path)
    assert loaded.get("fake-model", "chunk text") == RST_OUTPUT

    llm = FakeLLM()
    parser = RstParser(llm, cache=RstCache.load(cache_path), require_cache=True)
    tree = parser.parse_chunks([Document(page_content="chunk text", metadata={"chunk_id": 1})])[0]
    assert llm.calls == 0
    assert tree.edus and tree.relations

    with pytest.raises(MissingRstCache):
        RstParser(FakeLLM(), cache=RstCache(), require_cache=True).parse_chunks(
            [Document(page_content="uncached", metadata={"chunk_id": 1})]
        )


def test_online_parser_fills_cache_when_allowed(tmp_path):
    llm = FakeLLM()
    cache = RstCache()
    parser = RstParser(llm, cache=cache, require_cache=False)
    parser.parse_chunks([Document(page_content="new chunk", metadata={"chunk_id": 2})])
    assert llm.calls == 1
    assert cache.get("fake-model", "new chunk") is not None
    parser.parse_chunks([Document(page_content="new chunk", metadata={"chunk_id": 3})])
    assert llm.calls == 1
