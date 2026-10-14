"""Background refresh for the latest NWP issue files."""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config import get_settings

LOGGER = logging.getLogger("nwp.scheduler")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFRESH_HOURS = (2, 4, 9, 14, 16, 21)
NWP_SNAPSHOT = PROJECT_ROOT / "data" / "weather" / "latest_station_nwp.json"


def _refresh_hours() -> tuple[int, ...]:
    raw = get_settings().nwp_refresh_hours
    try:
        hours = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
        if not hours or any(hour < 0 or hour > 23 for hour in hours):
            raise ValueError
        return hours
    except (AttributeError, TypeError, ValueError):
        LOGGER.warning("invalid nwp_refresh_hours=%r; using defaults", raw)
        return DEFAULT_REFRESH_HOURS


def next_refresh(now: datetime | None = None) -> datetime:
    """Return the next configured Beijing-time refresh instant."""
    settings = get_settings()
    tz = ZoneInfo(settings.timezone_name)
    current = (now or datetime.now(tz)).astimezone(tz)
    for day_offset in range(3):
        day = (current + timedelta(days=day_offset)).date()
        for hour in _refresh_hours():
            candidate = datetime(day.year, day.month, day.day, hour, tzinfo=tz)
            if candidate > current:
                return candidate
    raise RuntimeError("unable to calculate next NWP refresh time")


def refresh_latest_nwp() -> None:
    """Run one download/crop cycle and raise on failure for logging."""
    settings = get_settings()
    script = PROJECT_ROOT / "scripts" / "extract_latest_nwp.py"
    command = [sys.executable, str(script), "--remote", settings.nwp_ssh_host, "--remote-dir", settings.nwp_remote_dir]
    LOGGER.info(
        "NWP refresh started: host=%s remote_dir=%s script=%s",
        settings.nwp_ssh_host,
        settings.nwp_remote_dir,
        script,
    )
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30 * 60, check=False)
    if result.returncode != 0:
        LOGGER.error("NWP refresh failed (exit=%s): %s", result.returncode, result.stderr[-4000:])
        raise RuntimeError(f"NWP refresh exited with {result.returncode}")
    LOGGER.info("NWP refresh completed: %s", result.stdout.strip()[-2000:])


def local_nwp_is_stale(max_age_hours: float = 12.0) -> bool:
    """Return True when the local snapshot is missing or older than the SLA."""
    if not NWP_SNAPSHOT.exists():
        return True
    try:
        payload = json.loads(NWP_SNAPSHOT.read_text(encoding="utf-8"))
        issued = datetime.fromisoformat(str(payload["issued_at_utc"]).replace("Z", "+00:00"))
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=ZoneInfo("UTC"))
        age = (datetime.now(ZoneInfo("UTC")) - issued.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600
        return age > max_age_hours
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return True


async def nwp_refresh_loop() -> None:
    """Wait for each Beijing-time slot and refresh without blocking the API."""
    settings = get_settings()
    LOGGER.info(
        "NWP scheduler started: host=%s remote_dir=%s refresh_hours=%s timezone=%s",
        settings.nwp_ssh_host,
        settings.nwp_remote_dir,
        _refresh_hours(),
        settings.timezone_name,
    )
    # Catch up once after a backend restart. This prevents a stale local file
    # from surviving until the next scheduled slot.
    if local_nwp_is_stale():
        LOGGER.warning("local NWP snapshot is stale or missing; refreshing immediately")
        try:
            await asyncio.to_thread(refresh_latest_nwp)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("startup NWP catch-up failed; scheduled slots remain active")
    while True:
        target = next_refresh()
        delay = max(1.0, (target - datetime.now(target.tzinfo)).total_seconds())
        LOGGER.info("next NWP refresh scheduled at %s", target.isoformat())
        await asyncio.sleep(delay)
        try:
            await asyncio.to_thread(refresh_latest_nwp)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("scheduled NWP refresh failed; next slot will be attempted")
