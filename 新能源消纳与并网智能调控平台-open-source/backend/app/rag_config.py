import os
from pathlib import Path


def load_backend_env() -> None:
    """Load non-secret environment entries from backend/.env if present.

    Explicit process environment variables always win. This keeps local
    development convenient while avoiding any key being committed to source.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def model_configured() -> bool:
    load_backend_env()
    return bool(os.getenv("POWER_LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")).strip())
