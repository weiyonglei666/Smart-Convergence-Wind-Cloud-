import pytest

from power_eval.config import PowerEvalConfig


def test_config_reads_model_and_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("POWER_LLM_MODEL", "DeepSeek-V4.1-Flash")
    cfg = PowerEvalConfig.from_env()
    assert cfg.api_key == "test-key"
    assert cfg.model == "DeepSeek-V4.1-Flash"
    assert cfg.api_model == "deepseek-flash"
    assert cfg.temperature == 0.0


def test_config_fails_with_actionable_error_when_key_is_missing(monkeypatch):
    monkeypatch.delenv("POWER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        PowerEvalConfig.from_env()


def test_config_supports_openai_compatible_local_server(monkeypatch):
    monkeypatch.setenv("POWER_LLM_API_KEY", "local-key")
    monkeypatch.setenv("POWER_LLM_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("POWER_LLM_MODEL", "Qwen3.8-27B")
    cfg = PowerEvalConfig.from_env()
    assert cfg.api_key == "local-key"
    assert cfg.base_url == "http://127.0.0.1:8001/v1"
    assert cfg.model == "Qwen3.8-27B"
    assert cfg.api_model == "Qwen3.8-27B"
