from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class StationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    station_id: str
    station_type: str
    name: str
    longitude: float
    latitude: float
    capacity_mw: float | None
    enabled: bool


class SnapshotResponse(BaseModel):
    station_id: str
    observed_at: datetime | None
    active_power_mw: float | None
    theory_power_mw: float | None
    available_power_mw: float | None
    limit_mark: int | None
    weather: dict[str, Any]
    freshness: str
    source: str
    quality: str


class PowerPointResponse(BaseModel):
    station_id: str
    observed_at: datetime
    active_power_mw: float | None
    theory_power_mw: float | None
    available_power_mw: float | None
