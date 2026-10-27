import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_latest_nwp import interpolate_rows


def test_nwp_rows_are_interpolated_to_15_minutes_and_keep_utc_plus_8():
    rows = [
        {"station_id": "FD0001", "latitude": 39.2, "longitude": 117.3, "valid_time_utc": "2026-01-01T00:00:00Z", "variables": {"t2m": 0.0}},
        {"station_id": "FD0001", "latitude": 39.2, "longitude": 117.3, "valid_time_utc": "2026-01-01T03:00:00Z", "variables": {"t2m": 3.0}},
    ]
    result = interpolate_rows(rows, step_minutes=15)
    assert len(result) == 13
    assert result[0]["valid_time_cn"].startswith("2026-01-01T08:00:00+08:00")
    assert abs(result[4]["variables"]["t2m"] - 1.0) < 1e-9
    assert result[-1]["valid_time_cn"].startswith("2026-01-01T11:00:00+08:00")
