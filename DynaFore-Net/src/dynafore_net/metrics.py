"""Station-capacity normalization and error-weighted real-time ACC."""
from __future__ import annotations

import numpy as np

def real_time_acc(y_true: np.ndarray, y_pred: np.ndarray, cap: float, horizon: int = 16) -> np.ndarray:
    """Official real-time ACC over the first `horizon` points.

    Returns a 0-100 percentage per sample. Negative ACC is intentionally not
    clipped because the provided rule does not require truncation.
    """
    yt = np.asarray(y_true, dtype=float)[..., :horizon]
    yp = np.asarray(y_pred, dtype=float)[..., :horizon]
    err = np.abs(yt - yp)
    denom = err.sum(axis=-1)
    sq = (yt - yp) ** 2
    weighted = np.zeros_like(denom, dtype=float)
    nonzero = denom > 0
    weighted[nonzero] = np.sqrt((sq[nonzero] * (err[nonzero] / denom[nonzero, None])).sum(axis=-1))
    return (1.0 - weighted / cap) * 100.0



def metrics(y: np.ndarray, pred: np.ndarray, cap: float) -> dict[str, float]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("capacity_mw must be positive and finite")
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != 16 or len(y) == 0:
        raise ValueError("Expected matching nonempty [samples, 16] arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite arrays on identical valid points")
    err = pred - y
    abs_err = np.abs(err)
    def r2(h: int) -> float:
        yt, ep = y[:, :h], err[:, :h]
        denom = ((yt - yt.mean()) ** 2).sum()
        return float(1.0 - (ep ** 2).sum() / denom) if denom > 0 else float("nan")
    ramp_true, ramp_pred = np.diff(y, axis=1), np.diff(pred, axis=1)
    event_true, event_pred = np.abs(ramp_true) >= 0.1 * cap, np.abs(ramp_pred) >= 0.1 * cap
    tp = float((event_true & event_pred).sum()); fp = float((~event_true & event_pred).sum()); fn = float((event_true & ~event_pred).sum())
    precision, recall = tp / max(tp + fp, 1.0), tp / max(tp + fn, 1.0)
    horizon = min(16, y.shape[1])
    return {
        "ACC@16": float(real_time_acc(y, pred, cap, horizon).mean()),
        "MSE@16": float((err[:, :horizon] ** 2).mean()),
        "MAE@16": float(abs_err[:, :horizon].mean()), "RMSE@16": float(np.sqrt((err[:, :horizon] ** 2).mean())),
        "nMAE@16": float(abs_err[:, :horizon].mean() / cap), "nRMSE@16": float(np.sqrt((err[:, :horizon] ** 2).mean()) / cap),
        "R2@16": r2(horizon),
        "ramp_F1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "ramp_direction_accuracy": float((np.sign(ramp_true) == np.sign(ramp_pred)).mean()),
    }
