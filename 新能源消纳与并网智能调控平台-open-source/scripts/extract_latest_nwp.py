"""Download the newest NWP files and extract station-level 15-minute series.

Spatial matching uses bilinear interpolation on the four surrounding grid
points (falling back to nearest only if the grid is not rectilinear).  The raw
ECMWF valid times are UTC; the extracted rows preserve both the UTC timestamp
and the UTC+8 timestamp used by the business UI.  Native forecast steps (for
example 3 h) are linearly interpolated onto the 15-minute grid.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr
import cfgrib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.nwp.file_selector import select_latest_files

CN_TZ = ZoneInfo("Asia/Shanghai")
CN_OFFSET = pd.Timedelta(hours=8)
STATIONS = {
    "FD0001": (117.3590, 39.2483), "FD0002": (117.6120, 39.5231), "FD0003": (117.8220, 39.5586),
    "FD0061": (117.5300, 39.4514), "FD0063": (115.8350, 37.1339), "FD0082": (122.1900, 40.6500),
    "FD0083": (122.2680, 41.7631), "FD0084": (122.7250, 41.8444), "FD0085": (122.5310, 41.5105),
    "GF0050": (100.296829, 36.057195), "GF0051": (117.52976111, 39.45136944),
}

def ssh_lines(host: str, remote_dir: str) -> list[str]:
    result = subprocess.run(["ssh", host, "find", remote_dir, "-maxdepth", "1", "-type", "f"], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()

def download(host: str, remote_path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never expose a partially transferred GRIB as the current source file.
    # A killed/restarted scheduler can otherwise see the old target and skip
    # the download forever.
    if target.exists():
        return
    partial = target.with_name(target.name + ".part")
    lock = target.with_name(target.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"download already in progress: {partial}") from exc
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(["scp", f"{host}:{remote_path}", str(partial)], check=True)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def _iso_utc(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def _station_point(ds, lat: float, lon: float, lat_name: str, lon_name: str):
    """Bilinear interpolation at the station location (nearest fallback)."""
    try:
        return ds.interp({lat_name: lat, lon_name: lon}, method="linear")
    except (ValueError, KeyError):
        return ds.sel({lat_name: lat, lon_name: lon}, method="nearest")


def _valid_time(point, step_index: int):
    if "valid_time" in point.coords and "step" in point["valid_time"].dims:
        return point["valid_time"].isel(step=step_index).values
    base = pd.Timestamp(point["time"].values)
    if "step" in point.coords:
        step = point["step"].values
        offset = pd.Timedelta(step[step_index] if getattr(step, "ndim", 0) else step)
        return (base + offset).to_datetime64()
    return base.to_datetime64()


def collect_rows(path: Path, stations: dict[str, tuple[float, float]]) -> list[dict]:
    """Read every GRIB message group and emit one row per station/valid time."""
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    merged: dict[tuple[str, str], dict] = {}
    for ds in datasets:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            continue
        variables = [name for name in ds.data_vars if name not in {lat_name, lon_name}]
        if not variables:
            continue
        step_count = int(ds.sizes.get("step", 1))
        for station_id, (lon, lat) in stations.items():
            point = _station_point(ds, lat, lon, lat_name, lon_name)
            for step_index in range(step_count):
                values = {
                    name: float(point[name].isel(step=step_index).values)
                    if "step" in point[name].dims else float(point[name].values.flat[0])
                    for name in variables if point[name].values.size
                }
                valid_at = _iso_utc(_valid_time(point, step_index))
                row = merged.setdefault(
                    (station_id, valid_at),
                    {"station_id": station_id, "latitude": lat, "longitude": lon,
                     "valid_time_utc": valid_at, "variables": {}},
                )
                row["variables"].update(values)
    return list(merged.values())


def interpolate_rows(rows: list[dict], step_minutes: int = 15) -> list[dict]:
    """Linearly interpolate each station's native NWP steps to 15 minutes."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["station_id"], []).append(row)
    result: list[dict] = []
    for station_id, station_rows in sorted(grouped.items()):
        names = sorted({name for row in station_rows for name in row["variables"]})
        frame = pd.DataFrame(
            [{**row["variables"], "valid_time": row["valid_time_utc"]} for row in station_rows]
        )
        frame.index = pd.to_datetime(frame.pop("valid_time"), utc=True)
        frame = frame.sort_index()
        frame = frame.groupby(level=0).first()
        if len(frame) > 1:
            target = pd.date_range(frame.index.min(), frame.index.max(), freq=f"{step_minutes}min")
            frame = frame.reindex(target).interpolate(method="time", limit_area="inside")
        for stamp, values in frame.iterrows():
            utc = _iso_utc(stamp)
            cn = pd.Timestamp(stamp).tz_convert(CN_TZ).isoformat()
            result.append({
                "station_id": station_id,
                "latitude": station_rows[0]["latitude"],
                "longitude": station_rows[0]["longitude"],
                "valid_time_utc": utc,
                "valid_time_cn": cn,
                "variables": {name: float(values[name]) for name in names if pd.notna(values[name])},
            })
    return result


def nearest_values(path: Path, stations: dict[str, tuple[float, float]]) -> list[dict]:
    """Backward-compatible name; emits bilinearly interpolated 15-minute rows."""
    return interpolate_rows(collect_rows(path, stations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=os.getenv("NWP_SSH_HOST", "nwp-host"))
    parser.add_argument("--remote-dir", default=os.getenv("NWP_REMOTE_DIR", "/path/to/nwp/48h"))
    parser.add_argument("--output", default="data/weather/latest_station_nwp.json")
    args = parser.parse_args()
    selection = select_latest_files(ssh_lines(args.remote, args.remote_dir))
    output_dir = Path(args.output).parent
    sfc_path = output_dir / Path(selection.sfc).name
    download(args.remote, selection.sfc, sfc_path)
    rows = nearest_values(sfc_path, STATIONS)
    issued = selection.issue_sfc_utc
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    issued_cn = issued.astimezone(CN_TZ)
    payload = {
        "issued_at_utc": _iso_utc(issued),
        "issued_at_cn": issued_cn.isoformat(),
        "source_file": str(sfc_path),
        "temporal_resolution_minutes": 15,
        "spatial_method": "bilinear",
        "stations": rows,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "stations": len(rows), "issued_at_utc": payload["issued_at_utc"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
