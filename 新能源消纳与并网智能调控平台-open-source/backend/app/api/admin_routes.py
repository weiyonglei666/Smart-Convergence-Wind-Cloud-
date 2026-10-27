from dataclasses import asdict
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.adapters.stations import DEFAULT_STATIONS
from app.db import mysql_connection
from app.domain.admin_models import ADMIN_STATE, AUDIT_EVENTS, record_station_change
from app.config import get_settings
from app.rag_config import load_backend_env

router = APIRouter(prefix='/api/admin')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
NWP_SNAPSHOT = PROJECT_ROOT / 'data' / 'weather' / 'latest_station_nwp.json'


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


def _mysql_probe() -> dict:
    settings = get_settings()
    if not settings.mysql_database or not settings.mysql_user:
        return {'status': 'incomplete', 'detail': '数据库名或账号未配置'}
    try:
        connection = mysql_connection(connect_timeout=3)
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
                cursor.fetchone()
        finally:
            connection.close()
        return {'status': 'connected', 'detail': f'{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}'}
    except Exception as exc:  # pragma: no cover - depends on deployment
        return {'status': 'error', 'detail': f'{type(exc).__name__}: {exc}'}


def _nwp_probe() -> dict:
    if not NWP_SNAPSHOT.exists():
        return {'status': 'missing', 'detail': '尚未生成本地 NWP 快照'}
    try:
        payload = json.loads(NWP_SNAPSHOT.read_text(encoding='utf-8'))
        issued = datetime.fromisoformat(str(payload['issued_at_utc']).replace('Z', '+00:00'))
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - issued).total_seconds() / 3600
        status = 'stale' if age_hours > 12 else 'connected'
        return {'status': status, 'detail': f'{len(payload.get("stations", []))} 行，起报 {payload.get("issued_at_cn") or payload.get("issued_at_utc")}，{age_hours:.1f} h 前'}
    except Exception as exc:
        return {'status': 'error', 'detail': f'{type(exc).__name__}: {exc}'}


def _llm_probe() -> dict:
    load_backend_env()
    base = os.getenv('POWER_LLM_BASE_URL', os.getenv('DEEPSEEK_BASE_URL', 'http://127.0.0.1:8002/v1')).rstrip('/')
    key = os.getenv('POWER_LLM_API_KEY', os.getenv('DEEPSEEK_API_KEY', '')).strip()
    if not base:
        return {'status': 'incomplete', 'detail': '生成服务地址未配置'}
    request = urllib.request.Request(base + '/models', headers={'Authorization': f'Bearer {key}'} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            json.loads(response.read().decode('utf-8'))
        return {'status': 'connected', 'detail': base}
    except Exception as exc:
        return {'status': 'unreachable', 'detail': f'{base} · {type(exc).__name__}'}


@router.get('/data-sources/status')
def data_source_status():
    mysql = _mysql_probe()
    nwp = _nwp_probe()
    return [
        {'name': 'mysql', 'label': 'MySQL 功率与预测库', **mysql},
        {'name': 'nwp-files', 'label': 'NWP 文件快照', **nwp},
    ]


@router.get('/models')
def models():
    llm = _llm_probe()
    return [
        {'name': 'wind-ultra-short', 'status': 'configured', 'detail': '由 DynaFore-Net 离线写入预测表'},
        {'name': 'pv-ultra-short', 'status': 'configured', 'detail': '由 DynaFore-Net 离线写入预测表'},
        {'name': 'knowledge-llm', 'status': llm['status'], 'detail': llm['detail']},
    ]


@router.post('/infrastructure/{server_id}/test')
def test_infrastructure(server_id: str):
    if server_id == 'mysql':
        result = _mysql_probe()
    elif server_id == 'nwp':
        settings = get_settings()
        try:
            completed = subprocess.run(
                ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', settings.nwp_ssh_host, 'true'],
                capture_output=True, text=True, timeout=8, check=False,
            )
            result = {'status': 'connected' if completed.returncode == 0 else 'error',
                      'detail': f'{settings.nwp_ssh_host}:{settings.nwp_ssh_port}' if completed.returncode == 0 else (completed.stderr[-300:] or f'exit {completed.returncode}')}
        except Exception as exc:
            result = {'status': 'error', 'detail': f'{type(exc).__name__}: {exc}'}
    else:
        raise HTTPException(status_code=404, detail='server not found')
    record_station_change(f'SERVER:{server_id}', {}, result, action='server.test')
    return {'id': server_id, **result}


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
    record_station_change(f'SERVER:{server_id}', {}, values, action='server.update')
    return infrastructure()['servers']


@router.get('/audit-logs')
def audit_logs():
    return [asdict(event) for event in AUDIT_EVENTS]
