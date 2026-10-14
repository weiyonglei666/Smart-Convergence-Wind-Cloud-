from dataclasses import dataclass


@dataclass(frozen=True)
class ConsumptionMetrics:
    utilization_rate: float | None
    curtailed_mwh: float | None
    curtailment_rate: float | None
    quality: str
    formula_version: str = "v1-candidate"


def calculate_consumption(
    actual_mwh: float | None,
    theory_mwh: float | None,
    available_mwh: float | None,
    limit_mark: int | None,
) -> ConsumptionMetrics:
    if actual_mwh is None or theory_mwh is None or available_mwh is None:
        return ConsumptionMetrics(None, None, None, "insufficient")
    utilization = actual_mwh / available_mwh if available_mwh > 0 else None
    curtailed = max(theory_mwh - actual_mwh, 0.0)
    rate = curtailed / theory_mwh if theory_mwh > 0 else None
    quality = "complete" if limit_mark is not None else "partial"
    return ConsumptionMetrics(utilization, curtailed, rate, quality)
