"""Download only the newest NWP files and extract nearest station grid values."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
import cfgrib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.nwp.file_selector import select_latest_files

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

def nearest_values(path: Path, stations: dict[str, tuple[float, float]]) -> list[dict]:
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    rows = []
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
            point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
            for step_index in range(step_count):
                values = {name: float(point[name].isel(step=step_index).values) if "step" in point[name].dims else float(point[name].values.flat[0]) for name in variables if point[name].values.size}
                valid_time = point["valid_time"].isel(step=step_index).values if "valid_time" in point.coords and "step" in point["valid_time"].dims else point["time"].values
                valid_at = str(valid_time).replace(" ", "T")
                if not valid_at.endswith("Z") and "+" not in valid_at:
                    valid_at += "Z"
                rows.append({"station_id": station_id, "latitude": lat, "longitude": lon, "valid_time_utc": valid_at, "variables": values})
    return rows

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
    payload = {"issued_at_utc": selection.issue_sfc_utc.isoformat(), "issued_at_cn": selection.issue_sfc_utc.astimezone().isoformat(), "source_file": str(sfc_path), "stations": rows}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "stations": len(rows), "issued_at_utc": selection.issue_sfc_utc.isoformat()}, ensure_ascii=False))

if __name__ == "__main__":
    main()
