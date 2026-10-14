from datetime import datetime
from zoneinfo import ZoneInfo
from collections.abc import Callable
from collections import defaultdict
from math import sqrt

from app.config import get_settings
from app.domain.models import PowerPoint, RealtimeSnapshot, Station


CN_TZ = ZoneInfo("Asia/Shanghai")


def _forecast_table(station_id: str, source_type: str) -> str:
    prefix = get_settings().forecast_table_prefix.strip().strip("_")
    namespace = f"{prefix}_" if prefix else ""
    return f"{namespace}{station_id}_zdata_{source_type}_power_cdq_result"

DEFAULT_STATIONS = [
    Station("FD0001", "wind", "FD0001", 117.3590, 39.2483, 30.0),
    Station("FD0002", "wind", "FD0002", 117.6120, 39.5231, 12.0),
    Station("FD0003", "wind", "FD0003", 117.8220, 39.5586, 20.0),
    Station("FD0061", "wind", "FD0061", 117.5300, 39.4514, 50.0),
    Station("FD0063", "wind", "FD0063", 115.8350, 37.1339, 100.0),
    Station("FD0082", "wind", "FD0082", 122.1900, 40.6500, 300.0),
    Station("FD0083", "wind", "FD0083", 122.2680, 41.7631, 600.0),
    Station("FD0084", "wind", "FD0084", 122.7250, 41.8444, 300.0),
    Station("FD0085", "wind", "FD0085", 122.5310, 41.5105, 50.0),
    Station("GF0050", "pv", "GF0050", 100.296829, 36.057195, 1000.0),
    Station("GF0051", "pv", "GF0051", 117.52976111, 39.45136944, 49.8),
]


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=CN_TZ) if value.tzinfo is None else value.astimezone(CN_TZ)


class StationRepository:
    def __init__(self, connection_factory: Callable[[], object], stations: list[Station] | None = None):
        self.connection_factory = connection_factory
        self.stations = {x.station_id: x for x in (stations or DEFAULT_STATIONS)}

    def list_stations(self) -> list[Station]:
        # The admin console controls the active asset scope.  Keep the
        # repository's canonical station catalog intact, but exclude stations
        # explicitly disabled by the admin state from operational dashboards,
        # aggregates, maps and assistant context.
        from app.domain.admin_models import ADMIN_STATE

        return [
            station
            for station in self.stations.values()
            if ADMIN_STATE.get(station.station_id, station.enabled)
        ]

    def latest_snapshot(self, station_id: str) -> RealtimeSnapshot | None:
        station = self.stations.get(station_id)
        if station is None:
            return None
        if station.station_type == "wind":
            table = f"zdata_{station_id}_wind_power_real_15min"
            query = f"SELECT FARM_ID, DATE_TIME, ACTIVE_POWER, LIMIT_MARK, THEORY_POWER, AVAILABLE_POWER, WS FROM `{table}` ORDER BY DATE_TIME DESC LIMIT 1"
        else:
            table = f"zdata_{station_id}_pv_power_real_15min"
            query = f"SELECT FARM_ID, DATE_TIME, ACTIVE_POWER, THEORY_POWER, AVAILABLE_POWER, LIMIT_MARK, GHI FROM `{table}` ORDER BY DATE_TIME DESC LIMIT 1"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
        finally:
            connection.close()
        if not row:
            return None
        if station.station_type == "wind":
            _, observed, active, limit_mark, theory, available, wind_speed = row
            weather = {"wind_speed_ms": wind_speed}
        else:
            _, observed, active, theory, available, limit_mark, ghi = row
            weather = {"ghi_wm2": ghi}
        # LIMIT_MARK is a binary operating state: 1 = curtailed, 0 = not
        # curtailed. Some source rows store the default 0 as SQL NULL; expose
        # the business state consistently instead of treating it as unknown.
        normalized_limit_mark = 1 if limit_mark == 1 else 0
        return RealtimeSnapshot(
            station_id=station_id,
            observed_at=_aware(observed),
            active_power_mw=active,
            theory_power_mw=theory,
            available_power_mw=available,
            limit_mark=normalized_limit_mark,
            weather=weather,
        )

    def power_series(self, station_id: str, start: datetime, end: datetime) -> list[PowerPoint]:
        station = self.stations.get(station_id)
        if station is None:
            return []
        table = f"zdata_{station_id}_{'wind' if station.station_type == 'wind' else 'pv'}_power_real_15min"
        query = f"SELECT DATE_TIME, ACTIVE_POWER, THEORY_POWER, AVAILABLE_POWER FROM `{table}` WHERE DATE_TIME >= %s AND DATE_TIME <= %s ORDER BY DATE_TIME"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                db_start = start.astimezone(CN_TZ).replace(tzinfo=None) if start.tzinfo else start
                db_end = end.astimezone(CN_TZ).replace(tzinfo=None) if end.tzinfo else end
                cursor.execute(query, (db_start, db_end))
                rows = cursor.fetchall()
        finally:
            connection.close()
        return [PowerPoint(station_id, _aware(row[0]), row[1], row[2], row[3]) for row in rows]

    def latest_weather(self, station_id: str) -> dict:
        station = self.stations.get(station_id)
        if station is None:
            return {}
        if station.station_type == "wind":
            table = f"zdata_{station_id}_wind_tower_15min"
            query = f"SELECT * FROM `{table}` ORDER BY DATE_TIME DESC LIMIT 1"
        else:
            table = f"zdata_{station_id}_qxz_15min"
            query = f"SELECT * FROM `{table}` ORDER BY DATE_TIME DESC LIMIT 1"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
                columns = [column[0] for column in cursor.description or []]
        finally:
            connection.close()
        if not row:
            return {}
        values = dict(zip(columns, row))
        def first(*names):
            for name in names:
                if name in values:
                    return values[name]
            return None
        observed = values.get("DATE_TIME")
        if station.station_type == "wind":
            weather = {"wind_speed_ms": first("WS_10", "AVG_WS_10", "WS"), "wind_direction_deg": first("WD_10", "AVG_WD_10", "WD"), "temperature_c": first("T_10", "T"), "humidity_pct": first("H_10", "H"), "pressure_hpa": first("P_10", "P")}
        else:
            weather = {"ghi_wm2": first("GHI"), "dni_wm2": first("DNI"), "dhi_wm2": first("DHI"), "gti_wm2": first("GTI"), "wind_speed_ms": first("WS"), "wind_direction_deg": first("WD"), "temperature_c": first("T"), "humidity_pct": first("H")}
        return {"observed_at": _aware(observed), **weather}

    def weather_series(self, station_id: str, start: datetime, end: datetime) -> list[dict]:
        station = self.stations.get(station_id)
        if station is None:
            return []
        table = f"zdata_{station_id}_{'wind_tower' if station.station_type == 'wind' else 'qxz'}_15min"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                # Weather tables store local China wall-clock timestamps. The
                # API window is UTC, so convert the bounds to UTC+8 before
                # removing tzinfo for the MySQL comparison. Otherwise the
                # chart lags by eight hours and hides the newest observations.
                db_start = start.astimezone(CN_TZ).replace(tzinfo=None) if start.tzinfo else start
                db_end = end.astimezone(CN_TZ).replace(tzinfo=None) if end.tzinfo else end
                cursor.execute(f"SELECT * FROM `{table}` WHERE DATE_TIME >= %s AND DATE_TIME <= %s ORDER BY DATE_TIME", (db_start, db_end))
                rows = cursor.fetchall(); columns = [column[0] for column in cursor.description or []]
        finally:
            connection.close()
        result = []
        for row in rows:
            values = dict(zip(columns, row)); pick = lambda *names: next((values[n] for n in names if n in values), None)
            result.append({"observed_at": _aware(values.get("DATE_TIME")), "wind_speed_ms": pick("WS_10", "AVG_WS_10", "WS"), "wind_direction_deg": pick("WD_10", "AVG_WD_10", "WD"), "temperature_c": pick("T_10", "T"), "ghi_wm2": pick("GHI"), "dni_wm2": pick("DNI"), "dhi_wm2": pick("DHI")})
        return result

    def forecast_series(self, station_id: str, start: datetime, end: datetime) -> list[dict]:
        station = self.stations.get(station_id)
        if station is None:
            return []
        prefix = "wind" if station.station_type == "wind" else "pv"
        forecast_table = _forecast_table(station_id, prefix)
        actual_table = f"zdata_{station_id}_{prefix}_power_real_15min"
        query = f"SELECT p.PRE_DATE_TIME, p.ACTIVE_POWER, a.ACTIVE_POWER FROM `{forecast_table}` p LEFT JOIN `{actual_table}` a ON a.DATE_TIME = p.PRE_DATE_TIME WHERE p.PRE_DATE_TIME >= %s AND p.PRE_DATE_TIME <= %s ORDER BY p.PRE_DATE_TIME"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (start.replace(tzinfo=None), end.replace(tzinfo=None)))
                rows = cursor.fetchall()
        finally:
            connection.close()
        return [{"station_id": station_id, "observed_at": _aware(row[0]), "predicted_power_mw": row[1], "actual_power_mw": row[2]} for row in rows]

    def latest_forecast_window(self, station_id: str, points: int = 16) -> dict:
        station = self.stations.get(station_id)
        if station is None:
            return {"station_id": station_id, "issue_at": None, "points": []}
        prefix = "wind" if station.station_type == "wind" else "pv"
        forecast_table = _forecast_table(station_id, prefix)
        actual_table = f"zdata_{station_id}_{prefix}_power_real_15min"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT DATA_DATE, DATA_TIME FROM `{forecast_table}` ORDER BY DATA_DATE DESC, DATA_TIME DESC LIMIT 1")
                run = cursor.fetchone()
                if not run:
                    return {"station_id": station_id, "issue_at": None, "points": []}
                run_date, run_time = run
                cursor.execute(
                    f"SELECT PRE_DATE_TIME, ACTIVE_POWER FROM `{forecast_table}` WHERE DATA_DATE=%s AND DATA_TIME=%s ORDER BY PRE_DATE_TIME LIMIT %s",
                    (run_date, run_time, points),
                )
                future = cursor.fetchall()
                first_target = future[0][0] if future else datetime.combine(run_date, datetime.min.time()) + run_time
                cursor.execute(
                    f"SELECT DATE_TIME, ACTIVE_POWER FROM `{actual_table}` WHERE DATE_TIME < %s ORDER BY DATE_TIME DESC LIMIT %s",
                    (first_target, points),
                )
                history = list(reversed(cursor.fetchall()))
                history_start = history[0][0] if history else first_target
                cursor.execute(
                    f"SELECT PRE_DATE_TIME, ACTIVE_POWER FROM `{forecast_table}` WHERE PRE_DATE_TIME >= %s AND PRE_DATE_TIME < %s ORDER BY DATA_DATE DESC, DATA_TIME DESC, PRE_DATE_TIME",
                    (history_start, first_target),
                )
                history_forecasts = cursor.fetchall()
        finally:
            connection.close()
        issue_at = datetime.combine(run_date, datetime.min.time()) + run_time
        latest_prediction_by_target = {}
        for target, value in history_forecasts:
            latest_prediction_by_target.setdefault(target, value)
        rows = [{"station_id": station_id, "observed_at": _aware(ts), "predicted_power_mw": latest_prediction_by_target.get(ts), "actual_power_mw": value, "phase": "history"} for ts, value in history]
        rows.extend({"station_id": station_id, "observed_at": _aware(ts), "predicted_power_mw": value, "actual_power_mw": None, "phase": "forecast"} for ts, value in future)
        return {"station_id": station_id, "issue_at": _aware(issue_at), "horizon_points": len(future), "horizon_hours": len(future) * 0.25, "points": rows}

    def realtime_acc(self, station_id: str, window_points: int = 16, max_runs: int = 96) -> dict:
        station = self.stations.get(station_id)
        if station is None or not station.capacity_mw:
            return {"station_id": station_id, "acc": None, "run_count": 0}
        prefix = "wind" if station.station_type == "wind" else "pv"
        forecast_table = _forecast_table(station_id, prefix)
        actual_table = f"zdata_{station_id}_{prefix}_power_real_15min"
        query = f"SELECT p.DATA_DATE,p.DATA_TIME,p.PRE_DATE_TIME,p.ACTIVE_POWER,a.ACTIVE_POWER FROM `{forecast_table}` p JOIN `{actual_table}` a ON a.DATE_TIME=p.PRE_DATE_TIME WHERE p.PRE_DATE_TIME<=NOW() ORDER BY p.DATA_DATE DESC,p.DATA_TIME DESC,p.PRE_DATE_TIME LIMIT %s"
        connection = self.connection_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (max_runs * 40,))
                raw = cursor.fetchall()
        finally:
            connection.close()
        runs = defaultdict(list)
        for run_date, run_time, target, predicted, actual in raw:
            runs[(run_date, run_time)].append((target, float(predicted), float(actual)))
        scores = []
        for values in runs.values():
            values = sorted(values)[:window_points]
            if len(values) != window_points:
                continue
            errors = [predicted - actual for _, predicted, actual in values]
            absolute_sum = sum(abs(error) for error in errors)
            weighted_rmse = 0.0 if absolute_sum == 0 else sqrt(sum(error * error * abs(error) / absolute_sum for error in errors))
            scores.append(max(0.0, 1.0 - weighted_rmse / station.capacity_mw))
        return {"station_id": station_id, "acc": round(sum(scores) / len(scores) * 100, 4) if scores else None, "run_count": len(scores), "window_points": window_points, "capacity_mw": station.capacity_mw}
