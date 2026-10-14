from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr


class ChatCloudLLM(BaseChatModel):
    """LangChain adapter for the project's OpenAI-compatible cloud client."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    model_name: str
    _client: Any = PrivateAttr()

    def __init__(self, client: Any, model_name: str):
        super().__init__(model_name=model_name)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "openai-compatible-cloud"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        payload = [{"role": self._role(message), "content": str(message.content)} for message in messages]
        text = self._client.generate(payload)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @staticmethod
    def _role(message: BaseMessage) -> str:
        return {"human": "user", "ai": "assistant", "system": "system"}.get(message.type, message.type)
