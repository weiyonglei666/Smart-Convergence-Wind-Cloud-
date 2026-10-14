from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PowerDocument:
    text: str
    source_file: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_powergridqa(root: str | Path) -> list[PowerDocument]:
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
