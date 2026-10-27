"""Documented multi-term training objective.

The report describes a composite loss rather than a single MSE: near-horizon
weighting, a ramp-difference constraint, consistency between the clean and
augmented views, frequency/statistics alignment, and a mechanism penalty for
infeasible power states.  This module keeps every term explicit and returns
the component values for logging.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossCoefficients:
    step_decay: float = 0.02
    ramp: float = 0.05
    consistency: float = 0.10
    frequency: float = 0.01
    statistics: float = 0.01
    mechanism: float = 0.05
    official: float = 0.15


def step_weights(horizon: int, decay: float, device, dtype) -> torch.Tensor:
    """Nearer forecast steps receive a larger weight (mean weight = 1)."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    positions = torch.arange(horizon, device=device, dtype=dtype)
    weights = torch.exp(-float(decay) * positions)
    return weights / weights.mean().clamp_min(1e-8)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target).square() * weights).mean()


def _ramp_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    return F.mse_loss(pred.diff(dim=1), target.diff(dim=1))


def _official(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable form of the error-weighted business ACC penalty."""
    err = (pred - target).abs()
    return torch.sqrt((err.pow(3).sum(-1) + 1e-10) / (err.sum(-1) + 1e-6)).mean()


def _frequency_alignment(clean: torch.Tensor, augmented: torch.Tensor) -> torch.Tensor:
    left = torch.fft.rfft(clean.float(), dim=1).abs()
    right = torch.fft.rfft(augmented.float(), dim=1).abs()
    return F.l1_loss(left, right)


def _statistics_alignment(clean: torch.Tensor, augmented: torch.Tensor) -> torch.Tensor:
    left_mean, right_mean = clean.mean(1), augmented.mean(1)
    left_std = clean.std(1, unbiased=False)
    right_std = augmented.std(1, unbiased=False)
    return (left_mean - right_mean).square().mean() + (left_std - right_std).square().mean()


def mechanism_penalty(aux: dict | None, energy_type: str) -> torch.Tensor:
    """Penalise raw outputs that violate the physical feasible set."""
    if not aux or "pre_clamp" not in aux:
        return torch.zeros((), dtype=torch.float32)
    raw = aux["pre_clamp"]
    low = F.relu(-raw)
    high = F.relu(raw - 1.0)
    if energy_type == "pv":
        daylight = aux.get("daylight")
        if daylight is None:
            daylight = (raw.detach() > 0).to(raw.dtype)
        violation = daylight * (low + high) + (1.0 - daylight) * raw.abs()
    else:
        violation = low + high
    return violation.mean()


def composite_loss(
    clean: torch.Tensor,
    target: torch.Tensor,
    *,
    augmented: torch.Tensor | None = None,
    augmented_target: torch.Tensor | None = None,
    aux: dict | None = None,
    energy_type: str = "wind",
    coefficients: LossCoefficients | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the total loss and its detached components.

    ``clean``/``target`` and ``augmented``/``augmented_target`` are already
    capacity-normalised.  When no augmented view is supplied the consistency,
    frequency and statistics terms are zero and the total reduces to the
    clean task loss plus the mechanism penalty.
    """
    coef = coefficients or LossCoefficients()
    horizon = clean.shape[1]
    weights = step_weights(horizon, coef.step_decay, clean.device, clean.dtype)
    task = _weighted_mse(clean, target, weights)
    official = _official(clean, target)
    ramp = _ramp_loss(clean, target)
    consistency = clean.new_zeros(())
    frequency = clean.new_zeros(())
    statistics = clean.new_zeros(())
    if augmented is not None:
        aug_target = target if augmented_target is None else augmented_target
        task = 0.5 * (task + _weighted_mse(augmented, aug_target, weights))
        official = 0.5 * (official + _official(augmented, aug_target))
        ramp = 0.5 * (ramp + _ramp_loss(augmented, aug_target))
        consistency = F.mse_loss(clean, augmented)
        frequency = _frequency_alignment(clean, augmented)
        statistics = _statistics_alignment(clean, augmented)
    mechanism = mechanism_penalty(aux, energy_type)
    if mechanism.device != clean.device:
        mechanism = mechanism.to(clean.device)
    total = (
        task
        + coef.official * official
        + coef.ramp * ramp
        + coef.consistency * consistency
        + coef.frequency * frequency
        + coef.statistics * statistics
        + coef.mechanism * mechanism
    )
    parts = {
        "task": float(task.detach()),
        "official": float(official.detach()),
        "ramp": float(ramp.detach()),
        "consistency": float(consistency.detach()),
        "frequency": float(frequency.detach()),
        "statistics": float(statistics.detach()),
        "mechanism": float(mechanism.detach()),
        "total": float(total.detach()),
    }
    return total, parts
