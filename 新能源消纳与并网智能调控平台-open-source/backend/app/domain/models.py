from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class Station:
    station_id: str
    station_type: Literal["wind", "pv"]
    name: str
    longitude: float
    latitude: float
    capacity_mw: float | None
    enabled: bool = True


@dataclass(frozen=True)
class RealtimeSnapshot:
    station_id: str
    observed_at: datetime
    active_power_mw: float | None
    theory_power_mw: float | None
    available_power_mw: float | None
    limit_mark: int | None
    weather: dict[str, Any] = field(default_factory=dict)
    freshness: str = "unknown"
    source: str = "mysql"


@dataclass(frozen=True)
class PowerPoint:
    station_id: str
    observed_at: datetime
    active_power_mw: float | None
    theory_power_mw: float | None
    available_power_mw: float | None
