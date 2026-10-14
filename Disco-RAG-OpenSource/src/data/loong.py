import os
from dataclasses import dataclass
from typing import List

from ..io_utils import read_jsonl


@dataclass
class LoongExample:
    id: str
    question: str
    answer: str
    task: str
    context: str
    length_set: str


def load_loong(root: str, split: str = "test") -> List[LoongExample]:
    rows = read_jsonl(os.path.join(root, f"{split}.jsonl"))
    examples = []
    for r in rows:
        examples.append(
            LoongExample(
                id=str(r["id"]),
                question=r["question"],
                answer=r["answer"],
                task=r["task"],
                context=r["context"],
                length_set=r["length_set"],
            )
        )
    return examples
