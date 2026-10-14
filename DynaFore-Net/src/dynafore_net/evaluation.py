"""Reload a best checkpoint and evaluate without refitting scalers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import load_station_arrays, split_indices
from .model import DynaForeNet
from .training import dynafore_flags, save_run


def evaluate_run(data_dir: Path, run: Path, device: str = "cpu", batch_size: int = 256) -> None:
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    meta, cfg = checkpoint["meta"], checkpoint["config"]
    x, future, y, origins, split, current_meta = load_station_arrays(data_dir, meta["station"])
    for key in ["station", "kind", "lookback", "pred_len", "capacity_mw", "x_columns", "future_columns"]:
        if current_meta[key] != meta[key]:
            raise ValueError(f"Checkpoint/data metadata mismatch: {key}")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
    model = DynaForeNet(x.shape[-1], future.shape[-1], meta["lookback"], meta["pred_len"], stats, cfg,
                       dynafore_flags(checkpoint["variant"], meta["kind"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    extra = {key: checkpoint[key] for key in ["variant", "auto_da_used", "best_epoch", "epochs_trained", "parameters", "seed"]}
    extra.update({"future_weather_used": True, "future_weather_source": "measured_future"})
    for name, indices in split_indices(split).items():
        if name == "train":
            continue
        predictions = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                j = indices[start:start + batch_size]
                xb = torch.as_tensor((x[j] - stats["x_mean"]) / stats["x_std"], device=device)
                fb = torch.as_tensor((future[j] - stats["f_mean"]) / stats["f_std"], device=device)
                predictions.append(model(xb, fb).cpu().numpy() * meta["capacity_mw"])
        if not predictions:
            raise ValueError(f"Empty {name} split")
        save_run(run, "DynaFore-Net", meta["station"], meta["capacity_mw"], y[indices],
                 np.concatenate(predictions), origins[indices], name, extra)
