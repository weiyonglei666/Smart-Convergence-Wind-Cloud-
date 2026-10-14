from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BenchItem:
    item_id: str
    question: str
    reference: str
    category: str
    split: str


def load_elecbench(root: str | Path) -> list[BenchItem]:
    items: list[BenchItem] = []
    for path in sorted(Path(root).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Some PowerGridQA JSONL files contain a plain-text title
                    # line before the records.
                    continue
                if not isinstance(row, dict):
                    continue
                question = str(row.get("input") or row.get("question") or "").strip()
                reference = str(row.get("output") or row.get("answer") or "").strip()
                if question:
                    items.append(BenchItem(f"{path.stem}:{line_no}", question, reference, str(row.get("index_class", "unknown")), path.stem))
    if not items:
        raise ValueError(f"No ElecBench JSONL records found under {root}")
    return items


def lexical_recall(answer: str, reference: str) -> float:
    ref = {token.lower() for token in reference.split() if token.strip()}
    if not ref:
        return 0.0
    return len(ref & {token.lower() for token in answer.split()}) / len(ref)


def run_resumable(items: list[BenchItem], answer: Callable[[str], object], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    previous = json.loads(line)
                    if not previous.get("error") and previous.get("answer"):
                        completed.add(previous["item_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    with output.open("a", encoding="utf-8") as handle:
        for item in items:
            if item.item_id in completed:
                continue
            started = time.perf_counter()
            try:
                result = answer(item.question)
                record = {"item_id": item.item_id, "question": item.question, "reference": item.reference, "category": item.category, "split": item.split, "answer": getattr(result, "answer", str(result)), "method": getattr(result, "method", "unknown"), "source_ids": getattr(result, "source_ids", []), "retrieval_scores": getattr(result, "retrieval_scores", []), "metadata": getattr(result, "metadata", {}), "latency_seconds": time.perf_counter() - started, "error": None}
            except Exception as exc:  # keep prior results resumable
                record = {"item_id": item.item_id, "question": item.question, "reference": item.reference, "category": item.category, "split": item.split, "answer": "", "method": "unknown", "source_ids": [], "latency_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def summarize_results(path: str | Path) -> dict[str, float]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in rows if not row.get("error")]
    return {"count": float(len(rows)), "success_rate": len(valid) / len(rows) if rows else 0.0, "mean_latency_seconds": sum(row.get("latency_seconds", 0.0) for row in valid) / len(valid) if valid else 0.0, "mean_lexical_recall": sum(lexical_recall(row.get("answer", ""), row.get("reference", "")) for row in valid) / len(valid) if valid else 0.0}
