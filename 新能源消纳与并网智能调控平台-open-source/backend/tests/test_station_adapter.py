from datetime import datetime

from app.adapters.stations import StationRepository


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.last_query = ""

    def execute(self, query, params=None):
        self.last_query = query

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return FakeCursor(self.rows)

    def close(self):
        pass


def test_station_repository_normalizes_wind_snapshot():
    rows = [("FD0001", datetime(2026, 9, 21, 20, 30), 5.013, 0, 4.88867, 4.88867, 4.59713)]
    repo = StationRepository(lambda: FakeConnection(rows))
    snapshot = repo.latest_snapshot("FD0001")
    assert snapshot is not None
    assert snapshot.station_id == "FD0001"
    assert snapshot.active_power_mw == 5.013
    assert snapshot.weather["wind_speed_ms"] == 4.59713


def test_station_repository_preserves_missing_pv_values():
    rows = [("GF0050", datetime(2026, 9, 21, 20, 30), -0.01, None, None, None, 0.0)]
    repo = StationRepository(lambda: FakeConnection(rows))
    snapshot = repo.latest_snapshot("GF0050")
    assert snapshot is not None
    assert snapshot.theory_power_mw is None
    assert snapshot.active_power_mw == -0.01
