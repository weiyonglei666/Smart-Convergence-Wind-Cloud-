import os
from dataclasses import dataclass
from typing import List

from ..io_utils import read_jsonl


@dataclass
class SciNewsExample:
    id: str
    article: str
    summary: str


def load_scinews(root: str, split: str = "test") -> List[SciNewsExample]:
    rows = read_jsonl(os.path.join(root, f"{split}.jsonl"))
    examples = []
    for i, r in enumerate(rows):
        examples.append(
            SciNewsExample(
                id=str(r.get("id", i)),
                article=r["article"],
                summary=r.get("lay_summary", r.get("summary", "")),
            )
        )
    return examples
