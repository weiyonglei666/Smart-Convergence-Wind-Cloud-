"""Small serialization helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(item) for k, item in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(item) for item in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
