from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWLEDGE_SUFFIXES = (".txt", ".md")


@dataclass(frozen=True)
class PowerDocument:
    text: str
    source_file: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_knowledge_corpus(root: str | Path) -> list[PowerDocument]:
    """Load the platform's private knowledge corpus (textbooks, reports, NERC notes).

    The PowerGridQA benchmark is an evaluation set, not a knowledge source.
    It is deliberately excluded here so reference answers can never be returned
    as retrieval evidence.
    """
    root = Path(root)
    documents: list[PowerDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in KNOWLEDGE_SUFFIXES:
            continue
        if path.name.lower().startswith("readme"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        relative = path.relative_to(root).as_posix()
        documents.append(PowerDocument(text, relative, relative, {"suffix": path.suffix.lower()}))
    return documents


def load_powergridqa(root: str | Path) -> list[PowerDocument]:
    """Legacy evaluation-fixture reader; never use it to build a retrieval index."""
    root = Path(root)
    documents: list[PowerDocument] = []
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                text_parts = [record.get(key, "") for key in ("question", "answer", "input", "output", "text")]
                text = "\n".join(str(part).strip() for part in text_parts if str(part).strip())
                if not text:
                    continue
                documents.append(PowerDocument(text, path.name, f"{path.name}:{line_no}", {"line": line_no}))
    if not documents:
        raise ValueError(f"No non-empty JSONL documents found under {root}")
    return documents
