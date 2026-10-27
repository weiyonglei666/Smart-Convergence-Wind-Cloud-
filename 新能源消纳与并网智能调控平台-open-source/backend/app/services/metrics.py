from dataclasses import dataclass


@dataclass(frozen=True)
class ConsumptionMetrics:
    utilization_rate: float | None
    curtailed_mw: float | None
    curtailment_rate: float | None
    quality: str
    formula_version: str = "v2-available-power"


def calculate_consumption(
    actual_mw: float | None,
    theory_mw: float | None,
    available_mw: float | None,
    limit_mark: int | None,
) -> ConsumptionMetrics:
    """Compute consumption with the documented available-power denominator.

    ``弃电功率 = max(0, 可利用功率 - 实际并网功率)`` and
    ``消纳率 = 实际功率 / 可利用功率``.  Theoretical power is retained only for
    quality/context checks; it must not replace a missing available-power
    field.  When actual or available power is missing the metric is kept
    missing instead of being back-filled.
    """
    if actual_mw is None or available_mw is None:
        return ConsumptionMetrics(None, None, None, "insufficient")
    utilization = actual_mw / available_mw if available_mw > 0 else None
    curtailed = max(available_mw - actual_mw, 0.0) if available_mw > 0 else None
    rate = curtailed / available_mw if curtailed is not None and available_mw > 0 else None
    quality = "complete" if limit_mark is not None and theory_mw is not None else "partial"
    return ConsumptionMetrics(utilization, curtailed, rate, quality)
