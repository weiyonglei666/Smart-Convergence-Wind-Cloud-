from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embedder import QwenEmbeddings
    from .llm import ChatVllmBeamSearch, build_llm

__all__ = ["ChatVllmBeamSearch", "build_llm", "QwenEmbeddings"]


def __getattr__(name):
    if name in ("ChatVllmBeamSearch", "build_llm"):
        from . import llm

        return getattr(llm, name)
    if name == "QwenEmbeddings":
        from .embedder import QwenEmbeddings

        return QwenEmbeddings
    raise AttributeError(name)
