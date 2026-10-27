from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_admin_station_update_writes_audit_event():
    response = client.patch('/api/admin/stations/FD0001', json={'enabled': False})
    assert response.status_code == 200
    assert response.json()['enabled'] is False
    logs = client.get('/api/admin/audit-logs').json()
    assert logs[-1]['station_id'] == 'FD0001'


def test_admin_rejects_unknown_station():
    response = client.patch('/api/admin/stations/UNKNOWN', json={'enabled': False})
    assert response.status_code == 404


def test_data_source_status_reports_missing_configuration():
    response = client.get('/api/admin/data-sources/status')
    assert response.status_code == 200
    payload = {item['name']: item for item in response.json()}
    assert payload['mysql']['status'] == 'incomplete'
    assert payload['nwp-files']['status'] in {'missing', 'stale', 'connected'}
