import os
from dataclasses import dataclass, field
from typing import Dict, List

from ..io_utils import read_json


@dataclass
class AsqaExample:
    id: str
    question: str
    qa_pairs: List[Dict] = field(default_factory=list)
    annotations: List[Dict] = field(default_factory=list)
    long_answers: List[str] = field(default_factory=list)
    short_answer_groups: List[List[str]] = field(default_factory=list)
    context: str = ""


def _build_context(qa_pairs: List[Dict]) -> str:
    parts = []
    for pair in qa_pairs:
        ctx = pair.get("context", "")
        if ctx:
            parts.append(ctx)
    return "\n\n".join(parts)


def load_asqa(root: str, split: str = "dev") -> List[AsqaExample]:
    data = read_json(os.path.join(root, f"{split}.json"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    examples = []
    items = data.items() if isinstance(data, dict) else enumerate(data)
    for key, item in items:
        qa_pairs = item.get("qa_pairs", [])
        annotations = item.get("annotations", [])
        long_answers = [a["long_answer"] for a in annotations if a.get("long_answer")]
        short_groups = [p.get("short_answers", []) for p in qa_pairs]
        examples.append(
            AsqaExample(
                id=str(item.get("sample_id", key)),
                question=item.get("ambiguous_question", item.get("question", "")),
                qa_pairs=qa_pairs,
                annotations=annotations,
                long_answers=long_answers,
                short_answer_groups=short_groups,
                context=item.get("context", "") or _build_context(qa_pairs),
            )
        )
    return examples
