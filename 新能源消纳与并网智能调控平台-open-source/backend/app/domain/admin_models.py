from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class AuditEvent:
    action: str
    station_id: str
    before: dict
    after: dict
    created_at: datetime


ADMIN_STATE: dict[str, bool] = {}
AUDIT_EVENTS: list[AuditEvent] = []


def record_station_change(station_id: str, before: dict, after: dict, action: str = 'station.update') -> AuditEvent:
    event = AuditEvent(action, station_id, before, after, datetime.now(timezone.utc))
    AUDIT_EVENTS.append(event)
    return event
