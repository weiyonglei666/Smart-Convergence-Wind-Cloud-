import os
import subprocess
import sys
from pathlib import Path

from power_eval.config import PowerEvalConfig


ROOT = Path(__file__).resolve().parents[1]


def test_release_uses_shared_budgets():
    config = PowerEvalConfig(api_key='test-only')
    assert config.top_k == 10
    assert config.max_tokens == 1024
    assert config.temperature == 0


def test_default_cli_uses_entire_benchmark_and_five_methods(tmp_path):
    corpus = tmp_path / 'corpus'
    benchmark = tmp_path / 'benchmark'
    corpus.mkdir()
    benchmark.mkdir()
    (corpus / 'fixture.jsonl').write_text('{"text":"Synthetic test passage"}\n')
    # Synthetic fixtures only: no copied benchmark records.
    (benchmark / 'fixture.jsonl').write_text(
        '\n'.join('{"question":"Synthetic question","answer":"Synthetic answer"}' for _ in range(205))
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(('POWER_', 'ELECBENCH_', 'DEEPSEEK_'))}
    env.update(POWER_CORPUS_ROOT=str(corpus), ELECBENCH_ROOT=str(benchmark))
    result = subprocess.run([sys.executable, 'scripts/run_power_eval.py', '--dry-run'],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert 'Evaluation questions: 205' in result.stdout
    assert 'disco_full, without_rst, without_graph, without_planning, standard_rag' in result.stdout
