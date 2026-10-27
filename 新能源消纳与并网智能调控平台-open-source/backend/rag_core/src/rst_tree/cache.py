"""Persistent offline cache for intra-chunk RST parses (platform RAG core)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class MissingRstCache(RuntimeError):
    pass


class RstCache:
    def __init__(self, entries: dict[str, str] | None = None, path: Path | None = None):
        self.entries = dict(entries or {})
        self.path = Path(path) if path is not None else None

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "RstCache":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload) if isinstance(payload, dict) else {}
        return cls({str(key): str(value) for key, value in entries.items()}, path=path)

    def get(self, model: str, text: str) -> str | None:
        return self.entries.get(self.key(model, text))

    def put(self, model: str, text: str, raw: str) -> None:
        self.entries[self.key(model, text)] = raw

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("RstCache.save requires a path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"format": "rst-cache-v1", "count": len(self.entries), "entries": self.entries}, ensure_ascii=False), encoding="utf-8")
        return target
