from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping

from .config import PowerEvalConfig


class DeepSeekClient:
    """Small OpenAI-compatible client with no secret persistence or logging."""

    def __init__(self, config: PowerEvalConfig):
        self.config = config

    def generate(self, messages: Iterable[Mapping[str, str]], max_tokens: int | None = None) -> str:
        payload = {
            "model": self.config.api_model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            # The evaluation records the assistant's final content. Disable
            # hidden reasoning so a short max_tokens budget is not exhausted
            # before the answer is emitted by Qwen3.8.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code < 500 or attempt == 2:
                    raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError(f"DeepSeek API connection failed: {exc.reason}") from exc
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"DeepSeek API request failed after retries: {last_error}")
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek API returned an invalid chat completion response") from exc
