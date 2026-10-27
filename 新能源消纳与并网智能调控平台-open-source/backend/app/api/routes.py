from dataclasses import asdict
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.adapters.stations import StationRepository
from app.api.schemas import PowerPointResponse, SnapshotResponse, StationResponse
from app.db import mysql_connection
from app.services.metrics import calculate_consumption
from app.services.time_alignment import CN_TZ, classify_freshness

router = APIRouter(prefix="/api")

# The real-time page has several widgets and two charts that ask for the same
# snapshot at once. Keep the latest MySQL aggregation briefly so one slow DB
# round-trip is shared by all of them.
_overview_cache: dict | None = None
_overview_cache_at = 0.0
_OVERVIEW_TTL_SECONDS = 15.0
_weather_cache: dict | None = None
_weather_cache_at = 0.0
_weather_series_cache: dict[tuple[str, int], tuple[float, list]] = {}
_consumption_cache: dict | None = None
_consumption_cache_at = 0.0


def get_repository() -> StationRepository:
    return StationRepository(mysql_connection)


Repo = Annotated[StationRepository, Depends(get_repository)]


def snapshot_response(snapshot) -> SnapshotResponse:
    if snapshot is None:
        raise HTTPException(status_code=200, detail="unavailable")
    now = datetime.now(timezone.utc)
    freshness = classify_freshness(snapshot.observed_at, now)
    quality = "complete" if snapshot.active_power_mw is not None else "insufficient"
    return SnapshotResponse(
        station_id=snapshot.station_id,
        observed_at=snapshot.observed_at,
        active_power_mw=snapshot.active_power_mw,
        theory_power_mw=snapshot.theory_power_mw,
        available_power_mw=snapshot.available_power_mw,
        limit_mark=snapshot.limit_mark,
        weather=snapshot.weather,
        freshness=freshness,
        source=snapshot.source,
        quality=quality,
    )


@router.get("/stations", response_model=list[StationResponse])
def stations(repo: Repo):
    return [StationResponse.model_validate(item) for item in repo.list_stations()]


@router.get("/stations/{station_id}/snapshot", response_model=SnapshotResponse)
def station_snapshot(station_id: str, repo: Repo):
    if station_id not in {x.station_id for x in repo.list_stations()}:
        raise HTTPException(status_code=404, detail="station not found")
    snapshot = repo.latest_snapshot(station_id)
    if snapshot is None:
        return SnapshotResponse(
            station_id=station_id, observed_at=None, active_power_mw=None,
            theory_power_mw=None, available_power_mw=None, limit_mark=None,
            weather={}, freshness="unavailable", source="mysql", quality="insufficient",
        )
    return snapshot_response(snapshot)


@router.get("/stations/{station_id}/power-series", response_model=list[PowerPointResponse])
def station_power_series(station_id: str, repo: Repo, hours: int = 24):
    if station_id not in {x.station_id for x in repo.list_stations()}:
        raise HTTPException(status_code=404, detail="station not found")
    bounded_hours = max(1, min(hours, 168))
    end = datetime.now(timezone.utc)
    points = repo.power_series(station_id, end - timedelta(hours=bounded_hours), end)
    return [PowerPointResponse.model_validate(point.__dict__) for point in points]


@router.get("/stations/{station_id}/forecast-series")
def station_forecast_series(station_id: str, repo: Repo, hours: int = 24):
    if station_id not in {x.station_id for x in repo.list_stations()}:
        raise HTTPException(status_code=404, detail="station not found")
    bounded_hours = max(1, min(hours, 168))
    end = datetime.now(timezone.utc)
    return repo.forecast_series(station_id, end - timedelta(hours=bounded_hours), end)


@router.get("/stations/{station_id}/forecast-window")
def station_forecast_window(station_id: str, repo: Repo, points: int = 16):
    return repo.latest_forecast_window(station_id, max(1, min(points, 40)))


@router.get("/stations/{station_id}/forecast-metrics")
def station_forecast_metrics(station_id: str, repo: Repo, hours: int = 24):
    points = station_forecast_series(station_id, repo, hours)
    pairs = [(item["predicted_power_mw"], item["actual_power_mw"]) for item in points if item["predicted_power_mw"] is not None and item["actual_power_mw"] is not None]
    if not pairs:
        return {"station_id": station_id, "sample_count": 0, "mae": None, "rmse": None}
    errors = [predicted - actual for predicted, actual in pairs]
    acc = repo.realtime_acc(station_id)
    return {"station_id": station_id, "sample_count": len(errors), "mae": round(sum(abs(error) for error in errors) / len(errors), 4), "rmse": round((sum(error * error for error in errors) / len(errors)) ** 0.5, 4), **acc}


@router.get("/overview/realtime")
def overview(repo: Repo):
    global _overview_cache, _overview_cache_at
    now_monotonic = time.monotonic()
    if _overview_cache is not None and now_monotonic - _overview_cache_at < _OVERVIEW_TTL_SECONDS:
        return _overview_cache
    # Each station is an independent read; parallel connections cut the first
    # dashboard load from 10+ serial round-trips to one bounded batch.
    def read_one(station):
        snapshot = repo.latest_snapshot(station.station_id)
        return snapshot_response(snapshot).model_dump(mode="json") if snapshot is not None else None
    with ThreadPoolExecutor(max_workers=6) as pool:
        snapshots = [item for item in pool.map(read_one, repo.list_stations()) if item is not None]
    _overview_cache = {"updated_at": datetime.now(timezone.utc), "stations": snapshots, "count": len(snapshots)}
    _overview_cache_at = now_monotonic
    return _overview_cache


@router.get("/overview/consumption")
def consumption_overview(repo: Repo):
    global _consumption_cache, _consumption_cache_at
    now_monotonic = time.monotonic()
    if _consumption_cache is not None and now_monotonic - _consumption_cache_at < 15:
        return _consumption_cache
    with ThreadPoolExecutor(max_workers=6) as pool:
        snapshots = list(pool.map(lambda station: repo.latest_snapshot(station.station_id), repo.list_stations()))
    snapshots = [item for item in snapshots if item is not None]
    actual = sum((item.active_power_mw or 0) for item in snapshots)
    theory = sum((item.theory_power_mw or 0) for item in snapshots)
    available = sum((item.available_power_mw or 0) for item in snapshots)
    per_station = {
        item.station_id: calculate_consumption(
            item.active_power_mw, item.theory_power_mw, item.available_power_mw, item.limit_mark,
        )
        for item in snapshots
    }
    # The documented consumption balance only needs actual and available
    # power.  Theory power is optional context and must not make a station
    # "incomplete" when the available-power fields are present.
    field_complete = [item for item in snapshots if per_station[item.station_id].utilization_rate is not None]
    # consumed (actual) + curtailed (available - actual) = available.  They
    # therefore use the same available-power denominator; theoretical power
    # remains available for quality checks and the analysis curves, but must
    # not be mixed into this pair of percentages.
    comparable = [
        item for item in field_complete
        if item.active_power_mw >= 0 and item.available_power_mw > 0 and item.active_power_mw <= item.available_power_mw
    ]
    comparable_actual = sum(item.active_power_mw for item in comparable)
    comparable_available = sum(item.available_power_mw for item in comparable)
    curtailed = sum(per_station[item.station_id].curtailed_mw for item in comparable)
    quality_ok = comparable_available > 0
    missing = [item.station_id for item in snapshots if item not in field_complete]
    anomalies = [item.station_id for item in field_complete if item not in comparable]
    _consumption_cache = {
        "updated_at": datetime.now(timezone.utc),
        "actual_power_mw": round(actual, 3),
        "theory_power_mw": round(theory, 3),
        "available_power_mw": round(available, 3),
        "consumption_rate": round(comparable_actual / comparable_available * 100, 2) if quality_ok else None,
        "curtailment_rate": round(curtailed / comparable_available * 100, 2) if quality_ok else None,
        "quality": "partial" if quality_ok and (missing or anomalies) else ("complete" if quality_ok else "anomaly"),
        "warning": (f"按 {len(comparable)}/{len(snapshots)} 个有效场站计算" + (f"；缺失：{', '.join(missing)}" if missing else "") + (f"；口径异常：{', '.join(anomalies)}" if anomalies else "")) if quality_ok else "有效场站不足，暂不计算消纳率",
        "station_breakdown": [{
            "station_id": item.station_id,
            "active_power_mw": item.active_power_mw,
            "theory_power_mw": item.theory_power_mw,
            "available_power_mw": item.available_power_mw,
            "consumption_rate": round(per_station[item.station_id].utilization_rate * 100, 2) if item in comparable else None,
            "curtailment_rate": round(per_station[item.station_id].curtailment_rate * 100, 2) if item in comparable and per_station[item.station_id].curtailment_rate is not None else None,
            "status": "complete" if item in comparable else ("missing_fields" if item not in field_complete else "anomaly"),
        } for item in snapshots],
        "comparable_station_count": len(comparable),
        "missing_station_ids": missing,
        "anomaly_station_ids": anomalies,
        "station_count": len(snapshots),
    }
    _consumption_cache_at = time.monotonic()
    return _consumption_cache


@router.get("/overview/weather-observation")
def weather_observation_overview(repo: Repo):
    global _weather_cache, _weather_cache_at
    if _weather_cache is not None and time.monotonic() - _weather_cache_at < 5:
        return _weather_cache
    def read_one(station):
        return {"station_id": station.station_id, "station_type": station.station_type, "weather": repo.latest_weather(station.station_id), "source": "wind_tower" if station.station_type == "wind" else "pv_irradiance"}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(read_one, repo.list_stations()))
    _weather_cache = {"updated_at": datetime.now(timezone.utc), "stations": rows}
    _weather_cache_at = time.monotonic()
    return _weather_cache


@router.get("/stations/{station_id}/weather-series")
def station_weather_series(station_id: str, repo: Repo, hours: int = 24):
    if station_id not in {x.station_id for x in repo.list_stations()}:
        raise HTTPException(status_code=404, detail="station not found")
    bounded_hours = max(1, min(hours, 168)); key = (station_id, bounded_hours); now = time.monotonic()
    cached = _weather_series_cache.get(key)
    if cached and now - cached[0] < 5:
        return cached[1]
    end = datetime.now(timezone.utc); result = repo.weather_series(station_id, end - timedelta(hours=bounded_hours), end)
    _weather_series_cache[key] = (now, result)
    return result


@router.get("/overview/nwp-latest")
def latest_nwp():
    path = Path(__file__).resolve().parents[3] / "data" / "weather" / "latest_station_nwp.json"
    if not path.exists():
        return {"available": False, "reason": "latest NWP extraction has not completed", "stations": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    issued_at = payload.get("issued_at_utc")
    age_hours = None
    if issued_at:
        try:
            issued_dt = datetime.fromisoformat(str(issued_at).replace("Z", "+00:00"))
            age_hours = round(max(0.0, (datetime.now(timezone.utc) - issued_dt).total_seconds() / 3600), 1)
        except ValueError:
            age_hours = None
    merged = {}
    for row in payload.get("stations", []):
        item = merged.setdefault(row["station_id"], {"station_id": row["station_id"], "latitude": row["latitude"], "longitude": row["longitude"], "variables": {}, "forecast": {}})
        item["variables"].update(row.get("variables", {}))
        valid = row.get("valid_time_utc")
        if valid:
            # JavaScript Date accepts milliseconds, while numpy may emit
            # nanosecond precision (nine fractional digits).
            if "." in valid:
                head, tail = valid.split(".", 1)
                valid = f"{head}.{tail[:3].rstrip('Z')}Z"
            valid_cn = row.get("valid_time_cn")
            if not valid_cn:
                valid_cn = (datetime.fromisoformat(valid.replace("Z", "+00:00")).astimezone(CN_TZ).isoformat())
            entry = item["forecast"].setdefault(valid, {"valid_time_utc": valid, "valid_time_cn": valid_cn, "variables": {}})
            entry["variables"].update(row.get("variables", {}))
    stations = []
    for item in merged.values():
        item["forecast"] = sorted(item["forecast"].values(), key=lambda point: point["valid_time_utc"])
        stations.append(item)
    return {"available": True, "issued_at_utc": payload.get("issued_at_utc"), "issued_at_cn": payload.get("issued_at_cn"), "source_file": payload.get("source_file"), "age_hours": age_hours, "freshness": "stale" if age_hours is not None and age_hours > 12 else "latest", "refresh_required": age_hours is not None and age_hours > 12, "stations": stations}


@router.get("/stream/realtime")
async def realtime_stream(request: Request, repo: Repo):
    async def events():
        while not await request.is_disconnected():
            payload = overview(repo)
            yield f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            await asyncio.sleep(5)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
