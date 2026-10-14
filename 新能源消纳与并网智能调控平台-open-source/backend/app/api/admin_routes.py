from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.adapters.stations import DEFAULT_STATIONS
from app.domain.admin_models import ADMIN_STATE, AUDIT_EVENTS, record_station_change
from app.config import get_settings

router = APIRouter(prefix='/api/admin')


class StationPatch(BaseModel):
    enabled: bool


class InfrastructurePatch(BaseModel):
    host: str | None = None
    port: int | None = None
    account: str | None = None
    database: str | None = None


@router.get('/stations')
def admin_stations():
    return [
        {**asdict(station), 'enabled': ADMIN_STATE.get(station.station_id, station.enabled)}
        for station in DEFAULT_STATIONS
    ]


@router.patch('/stations/{station_id}')
def update_station(station_id: str, patch: StationPatch):
    station = next((item for item in DEFAULT_STATIONS if item.station_id == station_id), None)
    if station is None:
        raise HTTPException(status_code=404, detail='station not found')
    before = {'enabled': ADMIN_STATE.get(station_id, station.enabled)}
    ADMIN_STATE[station_id] = patch.enabled
    after = {'enabled': patch.enabled}
    event = record_station_change(station_id, before, after)
    return {**asdict(station), 'enabled': patch.enabled}


@router.get('/data-sources/status')
def data_source_status():
    return [{'name': 'mysql', 'status': 'configured'}, {'name': 'nwp-files', 'status': 'latest-only'}]


@router.get('/infrastructure')
def infrastructure():
    """Return safe connection metadata for the admin console; never expose secrets."""
    settings = get_settings()
    overrides = getattr(infrastructure, '_overrides', {})
    servers = [
        {'id': 'mysql', 'name': '功率与预测数据库', 'kind': 'MySQL', 'host': settings.mysql_host, 'port': settings.mysql_port, 'account': settings.mysql_user or '未配置', 'database': settings.mysql_database or '未配置', 'status': 'configured' if settings.mysql_host and settings.mysql_database else 'incomplete', 'secret': '已配置' if settings.mysql_password else '未配置'},
        {'id': 'nwp', 'name': '气象 NWP 服务器', 'kind': 'SSH / 文件源', 'host': settings.nwp_ssh_host, 'port': settings.nwp_ssh_port, 'account': settings.nwp_ssh_user or '未配置', 'database': settings.nwp_remote_dir, 'status': 'configured' if settings.nwp_ssh_host and settings.nwp_remote_dir else 'incomplete', 'secret': '按 SSH Key 认证'},
    ]
    for server in servers:
        server.update(overrides.get(server['id'], {}))
    return {
        'servers': servers,
        'runtime': {'timezone': settings.timezone_name, 'realtime_interval_seconds': settings.realtime_interval_seconds},
    }


@router.patch('/infrastructure/{server_id}')
def update_infrastructure(server_id: str, patch: InfrastructurePatch):
    if server_id not in {'mysql', 'nwp'}:
        raise HTTPException(status_code=404, detail='server not found')
    values = {key: value for key, value in patch.model_dump().items() if value is not None}
    if 'port' in values and not 1 <= values['port'] <= 65535:
        raise HTTPException(status_code=422, detail='端口必须在 1-65535 范围内')
    overrides = getattr(infrastructure, '_overrides', {})
    current = dict(overrides.get(server_id, {})); current.update(values); overrides[server_id] = current; infrastructure._overrides = overrides
    record_station_change(f'SERVER:{server_id}', {}, values)
    return infrastructure()['servers']


@router.get('/models')
def models():
    return [{'name': 'wind-ultra-short', 'status': 'connected'}, {'name': 'pv-ultra-short', 'status': 'connected'}]


@router.get('/audit-logs')
def audit_logs():
    return [asdict(event) for event in AUDIT_EVENTS]
