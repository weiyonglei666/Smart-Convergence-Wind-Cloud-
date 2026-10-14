from power_eval.corpus import PowerDocument
from power_eval.graph import PowerKnowledgeGraph


def test_graph_retrieval_uses_entity_match_and_vector_fallback():
    docs = [
        PowerDocument("继电保护用于故障隔离", "a.jsonl", "a:1"),
        PowerDocument("变压器负责电压变换", "b.jsonl", "b:1"),
    ]
    graph = PowerKnowledgeGraph(docs)
    assert graph.retrieve("继电保护", 1)[0].document.source_id == "a:1"
    assert graph.retrieve("不存在的主题", 1)[0].document.source_id in {"a:1", "b:1"}
