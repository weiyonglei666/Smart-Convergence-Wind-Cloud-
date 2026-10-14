from dataclasses import dataclass
from datetime import datetime

from app.domain.models import Station


@dataclass(frozen=True)
class NwpPoint:
    issued_at_utc: datetime
    target_at_cn: datetime
    station_id: str
    variable: str
    value: float
    unit: str
    grid_distance_km: float
    source_file: str


class NwpUnavailable(RuntimeError):
    pass


class NwpService:
    """Station-level NWP contract; parser is injected so API tests stay local."""

    def __init__(self, extractor=None):
        self.extractor = extractor

    def station_forecast(self, station: Station, variables: list[str]) -> list[NwpPoint]:
        if self.extractor is None:
            raise NwpUnavailable("GRIB2 extractor is not configured")
        return self.extractor(station, variables)
