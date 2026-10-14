from datetime import datetime, timedelta, timezone

from app.services.time_alignment import classify_freshness, split_forecast_time


def test_split_forecast_time_keeps_issue_and_target_distinct():
    point = split_forecast_time("2026-09-21", "23:45:00", "2026-09-22T07:45:00")
    assert point.issued_at.isoformat() == "2026-09-21T23:45:00+08:00"
    assert point.target_at.isoformat() == "2026-09-22T07:45:00+08:00"
    assert point.horizon_minutes == 480


def test_classify_freshness_has_20_minute_realtime_boundary():
    now = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    assert classify_freshness(now - timedelta(minutes=20), now) == "realtime"
    assert classify_freshness(now - timedelta(minutes=21), now) == "near_realtime"
