import json

from power_eval.corpus import load_powergridqa
from power_eval.index import VectorIndex


def test_loader_preserves_source_and_skips_empty(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"question": "什么是继电保护", "answer": "用于故障隔离"}, ensure_ascii=False) + "\n" + json.dumps({"question": ""}) + "\n", encoding="utf-8")
    docs = load_powergridqa(tmp_path)
    assert len(docs) == 1
    assert docs[0].source_id == "sample.jsonl:1"
    assert VectorIndex(docs).search("继电保护", 1)[0].document.source_id == docs[0].source_id
