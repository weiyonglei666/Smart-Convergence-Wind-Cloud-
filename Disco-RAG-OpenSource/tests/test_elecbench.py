import json

from power_eval.elecbench import load_elecbench, run_resumable, summarize_results


def test_loader_and_resumable_runner(tmp_path):
    (tmp_path / "general.jsonl").write_text(json.dumps({"input": "问题", "output": "答案", "index_class": "reasoning"}) + "\n", encoding="utf-8")
    items = load_elecbench(tmp_path)
    output = tmp_path / "results.jsonl"
    run_resumable(items, lambda question: "答案", output)
    run_resumable(items, lambda question: "不会再次调用", output)
    assert summarize_results(output)["count"] == 1
    assert summarize_results(output)["success_rate"] == 1.0


def test_resumable_runner_retries_failed_records(tmp_path):
    (tmp_path / "general.jsonl").write_text(json.dumps({"input": "问题", "output": "答案"}) + "\n", encoding="utf-8")
    item = load_elecbench(tmp_path)[0]
    output = tmp_path / "retry.jsonl"
    run_resumable([item], lambda question: (_ for _ in ()).throw(RuntimeError("temporary")), output)
    run_resumable([item], lambda question: "答案", output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[-1]["error"] is None
