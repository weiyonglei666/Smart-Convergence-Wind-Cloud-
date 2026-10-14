from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class ForecastPoint:
    issued_at: datetime
    target_at: datetime
    horizon_minutes: int


def _parse_local(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return (parsed.replace(tzinfo=CN_TZ) if parsed.tzinfo is None else parsed.astimezone(CN_TZ))


def split_forecast_time(data_date: str, data_time: str, pre_date_time: str) -> ForecastPoint:
    issue_date = datetime.fromisoformat(data_date).date()
    issue_time = time.fromisoformat(data_time)
    issued_at = datetime.combine(issue_date, issue_time, tzinfo=CN_TZ)
    target_at = _parse_local(pre_date_time)
    return ForecastPoint(issued_at, target_at, round((target_at - issued_at).total_seconds() / 60))


Freshness = Literal["realtime", "near_realtime", "delayed", "offline", "invalid"]


def classify_freshness(observed_at: datetime, now: datetime) -> Freshness:
    observed = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=CN_TZ)
    current = now if now.tzinfo else now.replace(tzinfo=CN_TZ)
    age_minutes = (current.astimezone(CN_TZ) - observed.astimezone(CN_TZ)).total_seconds() / 60
    # A source can publish the next 15-minute slice slightly ahead of the
    # local machine clock. Treat a small clock/data-window skew as live data;
    # only materially future timestamps are invalid.
    if age_minutes < -20:
        return "invalid"
    if age_minutes <= 20:
        return "realtime"
    if age_minutes <= 60:
        return "near_realtime"
    if age_minutes <= 360:
        return "delayed"
    return "offline"
