"""Five equal-budget configurations for the power-domain adaptation."""
from .answerers import AnswerResult
from .cloud_llm import ChatCloudLLM
from .disco_pipeline import CloudDiscoRAGPipeline
from src.generation.standard_rag import StandardRagGenerator
from langchain_core.documents import Document

METHODS = ['disco_full', 'without_rst', 'without_graph', 'without_planning', 'standard_rag']

class AblationAnswerer:
    def __init__(self, client, index, method, rst_cache=None, require_rst_cache=False):
        if method not in METHODS:
            raise ValueError(method)
        self.client, self.index, self.method = client, index, method
        self.top_k = getattr(getattr(client, 'config', None), 'top_k', 10)
        self.llm = ChatCloudLLM(client, client.config.model, output_language='English')
        self.pipeline = CloudDiscoRAGPipeline(index, self.llm, top_k=self.top_k,
            use_rst=method != 'without_rst', use_graph=method != 'without_graph',
            use_planning=method != 'without_planning', rst_cache=rst_cache,
            require_rst_cache=require_rst_cache)

    def answer(self, question):
        if self.method in METHODS[:4]:
            r = self.pipeline.run(question)
            return AnswerResult(r.answer, self.method, r.source_ids, [],
                {'rst_enabled': str(self.pipeline.use_rst),
                 'graph_enabled': str(self.pipeline.use_graph),
                 'planning_enabled': str(self.pipeline.use_planning),
                 'plan': r.plan.serialize() if r.plan else 'None'})
        hits = self.index.search(question, self.top_k)
        chunks = [Document(page_content=h.document.text) for h in hits]
        answer = StandardRagGenerator(self.llm).generate(question, chunks)
        return AnswerResult(answer, self.method, [h.document.source_id for h in hits],
                            [h.score for h in hits], {'retrieval_query': question})
