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
    """Load the retrieval corpus (textbooks, technical reports, NERC notes).

    The evaluation benchmark is deliberately *not* read here: benchmark
    questions/answers must never become retrieval evidence.  A directory may
    contain nested folders; the relative path is kept as source metadata.
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
    if not documents:
        raise ValueError(
            f"No knowledge documents found under {root}; put UTF-8 .txt/.md corpus files there "
            "(do not put the evaluation JSONL in the corpus directory)"
        )
    return documents


def load_powergridqa(root: str | Path) -> list[PowerDocument]:
    """Legacy JSONL reader kept for evaluation-only fixtures.

    This reader includes both question and answer fields.  Do not use it to
    build a retrieval index; use :func:`load_knowledge_corpus` instead.
    """
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
                    # Some PowerGridQA files begin with a plain-text title line.
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
