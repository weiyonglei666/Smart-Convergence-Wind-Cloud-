from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.adapters.stations import StationRepository
from app.domain.models import RealtimeSnapshot
from app.main import app
from app.api.routes import get_repository


class FakeRepo:
    def list_stations(self):
        return StationRepository(lambda: None).list_stations()

    def latest_snapshot(self, station_id):
        if station_id == "FD0001":
            return RealtimeSnapshot("FD0001", datetime(2026, 9, 21, 12, tzinfo=timezone.utc), 5, 6, 6, 0)
        return None


app.dependency_overrides[get_repository] = lambda: FakeRepo()
client = TestClient(app)


def test_api_lists_configured_stations():
    response = client.get("/api/stations")
    assert response.status_code == 200
    assert len(response.json()) == 11


def test_api_returns_snapshot_for_known_station():
    response = client.get("/api/stations/FD0001/snapshot")
    assert response.status_code == 200
    assert response.json()["station_id"] == "FD0001"


def test_api_returns_404_for_unknown_or_empty_snapshot():
    response = client.get("/api/stations/UNKNOWN/snapshot")
    assert response.status_code == 404
    response = client.get("/api/stations/GF0050/snapshot")
    assert response.status_code == 200
    assert response.json()["freshness"] in {"offline", "unavailable"}


def test_assistant_status_reports_the_five_agents():
    response = client.get("/api/assistant/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["agents"] == ["问题理解智能体", "实时数据智能体", "预测分析智能体", "知识检索智能体", "报告生成智能体"]
    assert payload["document_count"] == 0


def test_consumption_uses_available_power_for_curtailment():
    response = client.get("/api/overview/consumption")
    assert response.status_code == 200
    payload = response.json()
    assert payload["consumption_rate"] == round(5 / 6 * 100, 2)
    assert payload["curtailment_rate"] == round(1 / 6 * 100, 2)
    assert payload["station_breakdown"][0]["status"] == "complete"
