from power_eval.cloud_llm import ChatCloudLLM
from power_eval.corpus import PowerDocument
from power_eval.disco_pipeline import CloudDiscoRAGPipeline
from power_eval.index import VectorIndex
import pytest
from unittest.mock import Mock


class StructuredFakeClient:
    def generate(self, messages, max_tokens=None):
        system = messages[0]["content"]
        if "Rhetorical Structure Theory" in system:
            return "EDUs:\n[1] Relay protection isolates faults.\n[2] It improves reliability.\nRST ANALYSIS:\nRELATION(EDU_1, EDU_2): {RESULT}\nTREE STRUCTURE:\nNUCLEUS[1]\nSATELLITE[2]"
        if "directed rhetorical graph" in system:
            return "CHUNK[1] -> CHUNK[2]: {ELABORATION}\nCHUNK[2] -> CHUNK[1]: {EVIDENCE}"
        if "produce a discourse-aware plan" in system:
            return "PLAN: Define protection, then explain its reliability effect."
        return "ANSWER: Relay protection isolates faults and improves reliability."


def test_cloud_pipeline_runs_official_rst_graph_plan_and_generation():
    docs = [
        PowerDocument("Relay protection isolates faults.", "a", "a:1"),
        PowerDocument("Fault isolation improves reliability.", "b", "b:1"),
    ]
    pipeline = CloudDiscoRAGPipeline(VectorIndex(docs), ChatCloudLLM(StructuredFakeClient(), "test-model"), top_k=2)
    result = pipeline.run("Why is relay protection important?")
    assert "isolates faults" in result.answer
    assert len(result.trees) == 2
    assert result.graph.num_chunks == 2
    assert result.plan.text.startswith("Define protection")
    assert result.source_ids == ["a:1", "b:1"]


@pytest.mark.parametrize('flag,module,operation', [
    ('use_rst', 'rst_parser', 'parse_chunks'),
    ('use_graph', 'graph_builder', 'build'),
    ('use_planning', 'planner', 'plan'),
])
def test_disabled_module_is_not_called(flag, module, operation):
    docs = [PowerDocument('Relay protection isolates faults.', 'a', 'a:1')]
    pipeline = CloudDiscoRAGPipeline(VectorIndex(docs),
        ChatCloudLLM(StructuredFakeClient(), 'test'), **{flag: False})
    disabled = Mock(side_effect=AssertionError('Disabled module was called'))
    setattr(getattr(pipeline, module), operation, disabled)
    result = pipeline.run('Why protection?')
    assert result.answer
    disabled.assert_not_called()


def test_disabled_structures_are_passed_as_none_to_generation():
    docs = [PowerDocument('Relay protection isolates faults.', 'a', 'a:1')]
    pipeline = CloudDiscoRAGPipeline(VectorIndex(docs),
        ChatCloudLLM(StructuredFakeClient(), 'test'), use_graph=False, use_planning=False)
    captured = {}
    original = pipeline.generator.generate
    def capture(query, chunks, trees=None, graph=None, plan=None):
        captured['graph'], captured['plan'] = graph, plan
        return original(query, chunks, trees, graph, plan)
    pipeline.generator.generate = capture
    pipeline.run('Why protection?')
    assert captured['graph'] is None
    assert captured['plan'] is None
