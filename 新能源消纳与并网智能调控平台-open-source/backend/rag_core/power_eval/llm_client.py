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

    def generate(self, messages: Iterable[Mapping[str, str]]) -> str:
        payload = {
            "model": self.config.api_model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
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
            choice = data["choices"][0]
            content = str(choice["message"].get("content") or "").strip()
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek API returned an invalid chat completion response") from exc
        if content and finish_reason == "length":
            # Continue once when the provider cut the visible answer at its
            # token limit. The continuation is explicitly requested without
            # reopening hidden reasoning.
            continuation_payload = dict(payload)
            continuation_payload["messages"] = [
                *payload["messages"],
                {"role": "assistant", "content": content},
                {"role": "user", "content": "上一条回答在此处被截断。请只从中断位置继续输出剩余的最终答案，不要重复前文，不要输出思考过程。"},
            ]
            continuation_request = urllib.request.Request(
                f"{self.config.base_url}/chat/completions",
                data=json.dumps(continuation_payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(continuation_request, timeout=self.config.timeout_seconds) as response:
                    continuation_data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                continuation = str(continuation_data["choices"][0]["message"].get("content") or "").strip()
                if continuation:
                    return f"{content}\n{continuation}"
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                pass
        if content:
            return content

        # Reasoning-capable models can occasionally spend the whole response
        # budget on hidden reasoning and leave the visible content empty. Retry
        # once with an explicit concise-answer instruction instead of exposing
        # reasoning_content or returning a blank message to the operator.
        retry_payload = dict(payload)
        retry_payload["messages"] = [
            *payload["messages"],
            {"role": "user", "content": "请停止展开内部推理，直接输出面向调度员的最终结论。使用简洁章节：现状、证据、建议、约束、验证指标。不要输出思考过程。"},
        ]
        retry_request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(retry_payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(retry_request, timeout=self.config.timeout_seconds) as response:
                retry_data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return str(retry_data["choices"][0]["message"].get("content") or "").strip()
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return ""
