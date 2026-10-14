from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr
from vllm import LLM as VLLM
from vllm.inputs import TextPrompt
from vllm.sampling_params import BeamSearchParams

from ..config import ModelConfig, RunConfig


class ChatVllmBeamSearch(BaseChatModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_path: str
    max_model_len: int
    beam_width: int = 3
    max_new_tokens: int = 1024

    _engine: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, model_cfg: ModelConfig, beam_width: int = 3, max_new_tokens: int = 1024):
        super().__init__(
            model_name=model_cfg.name,
            model_path=model_cfg.path,
            max_model_len=model_cfg.max_model_len,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
        )
        self._engine = VLLM(
            model=model_cfg.path,
            dtype=model_cfg.dtype,
            tensor_parallel_size=model_cfg.tensor_parallel_size,
            max_model_len=model_cfg.max_model_len,
            enforce_eager=False,
            gpu_memory_utilization=model_cfg.gpu_memory_utilization,
        )
        self._tokenizer = self._engine.get_tokenizer()

    @property
    def _llm_type(self) -> str:
        return "vllm-beam-search"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "beam_width": self.beam_width}

    def _render(self, messages: List[BaseMessage]) -> str:
        return self._tokenizer.apply_chat_template(
            convert_to_openai_messages(messages), tokenize=False, add_generation_prompt=True
        )

    def _beam_search(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        params = BeamSearchParams(beam_width=self.beam_width, max_tokens=max_new_tokens or self.max_new_tokens)
        outputs = self._engine.beam_search([TextPrompt(prompt=p) for p in prompts], params)
        return [o.sequences[0].text.strip() for o in outputs]

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self._beam_search([self._render(messages)], kwargs.get("max_new_tokens"))[0]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def batch(self, inputs: List, config=None, *, return_exceptions: bool = False, **kwargs) -> List[AIMessage]:
        prompts = [self._render(self._convert_input(i).to_messages()) for i in inputs]
        texts = self._beam_search(prompts, kwargs.get("max_new_tokens"))
        return [AIMessage(content=t) for t in texts]

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self._tokenizer.decode(ids[:max_tokens])


_CACHE: Dict[str, ChatVllmBeamSearch] = {}


def build_llm(run_cfg: RunConfig, backbone: Optional[str] = None) -> ChatVllmBeamSearch:
    key = backbone or run_cfg.backbone
    if key not in _CACHE:
        _CACHE[key] = ChatVllmBeamSearch(
            run_cfg.model(key),
            beam_width=run_cfg.decoding.beam_width,
            max_new_tokens=run_cfg.decoding.max_new_tokens,
        )
    return _CACHE[key]


def release_llms() -> None:
    import gc

    import torch

    _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
