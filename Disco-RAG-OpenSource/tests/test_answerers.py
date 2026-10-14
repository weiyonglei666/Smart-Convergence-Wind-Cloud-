from power_eval.answerers import DirectAnswerer, GraphRAGAnswerer, StandardRAGAnswerer
from power_eval.corpus import PowerDocument
from power_eval.graph import PowerKnowledgeGraph
from power_eval.index import VectorIndex


class FakeGenerator:
    def generate(self, messages):
        return messages[-1]["content"][:20]


def test_answerers_share_result_schema_and_graph_sources():
    docs = [PowerDocument("继电保护用于故障隔离", "a", "a:1")]
    index = VectorIndex(docs)
    results = [DirectAnswerer(FakeGenerator(), index).answer("继电保护"), StandardRAGAnswerer(FakeGenerator(), index).answer("继电保护"), GraphRAGAnswerer(FakeGenerator(), PowerKnowledgeGraph(docs, index)).answer("继电保护")]
    assert [result.method for result in results] == ["direct", "standard_rag", "graphrag"]
    assert results[2].source_ids == ["a:1"]
