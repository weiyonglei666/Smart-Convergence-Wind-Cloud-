"""ECMWF-style NWP ingestion for the documented DynaFore-Net input path.

The report specifies that the future-condition branch is driven by NWP, not by
future measured observations.  This module implements the documented release
contract:

* raw issue/valid times are UTC;
* issue and valid times are converted to Beijing time (UTC+8) before they are
  matched with the 15-minute power timeline;
* a forecast window may only use an issue whose ``ISSUE_TIME`` is not later
  than the forecast origin (no future-information leakage);
* a 3-hourly (or other native-step) issue is linearly interpolated onto the
  15-minute grid;
* exactly ten NWP variables plus two wind components and six calendar
  features form the 18-dimensional future condition vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CN_OFFSET = pd.Timedelta(hours=8)
NWP_META = {
    "ISSUE_TIME", "DATE_TIME", "TIME", "STATION_ID", "FARM_ID", "LAT", "LON",
    "LATITUDE", "LONGITUDE", "STEP", "LEAD_HOURS",
}
WIND_NWP_PREFERRED = [
    "WS", "WS_100", "WS_HUBHEIGHT", "WD", "WD_100", "WD_HUBHEIGHT",
    "T", "T2M", "H", "RH", "P", "SP", "CLOUD", "TCC", "TP", "PRECIP",
]
PV_NWP_PREFERRED = [
    "GHI", "GTI", "DNI", "DHI", "T", "T2M", "H", "RH", "P", "SP",
    "CLOUD", "TCC", "WS", "TP", "PRECIP",
]
# Common ECMWF / provider names accepted by the station-level loader.
ALIASES: dict[str, tuple[str, ...]] = {
    "GHI": ("GHI", "ghi", "ssrd", "ssr", "dswrf"),
    "GTI": ("GTI", "gti", "ssrd", "ssr"),
    "DNI": ("DNI", "dni", "dirswrf"),
    "DHI": ("DHI", "dhi", "difswrf"),
    "WS": ("WS", "ws", "WS_10", "ws10", "wind_speed", "wind_speed_ms"),
    "WD": ("WD", "wd", "WD_10", "wd10", "wind_direction", "wind_direction_deg"),
    "T": ("T", "t", "T2M", "t2m", "temperature"),
    "H": ("H", "h", "RH", "rh", "r", "humidity"),
    "P": ("P", "p", "SP", "sp", "msl", "pressure"),
    "CLOUD": ("CLOUD", "cloud", "TCC", "tcc", "cloud_cover"),
    "TP": ("TP", "tp", "PRECIP", "precip", "precipitation"),
}


@dataclass(frozen=True)
class NwpIssue:
    """One NWP issue on the Beijing-time 15-minute grid."""

    issue_at_cn: pd.Timestamp
    frame: pd.DataFrame


class NwpInputError(ValueError):
    """Raised when the supplied NWP file cannot satisfy the documented contract."""


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _pick_variables(columns: list[str], kind: str) -> list[str]:
    preferred = WIND_NWP_PREFERRED if kind == "wind" else PV_NWP_PREFERRED
    chosen = [name for name in preferred if name in columns]
    required = ("WS", "WD") if kind == "wind" else ("GHI", "GTI")
    for name in required:
        if name not in chosen:
            raise NwpInputError(f"{kind} NWP must provide {name}")
    # Keep the contract deterministic: ten physical NWP variables per kind.
    for name in sorted(columns):
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= 10:
            break
    if len(chosen) < 10:
        raise NwpInputError(
            f"NWP file exposes {len(chosen)} usable variables; the documented model "
            "contract requires exactly 10 physical variables (plus wind components)"
        )
    return chosen[:10]


def _canonicalize(raw: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Map common provider/ECMWF names onto the documented feature names."""
    values: dict[str, pd.Series] = {}
    used: set[str] = set()
    for canonical, aliases in ALIASES.items():
        source = next((name for name in aliases if name in numeric), None)
        if source is not None:
            values[canonical] = pd.to_numeric(raw[source], errors="coerce")
            used.add(source)
    # Derive wind speed/direction from 10 m components when necessary.
    if ("WS" not in values or "WD" not in values) and {"u10", "v10"} <= set(numeric):
        u = pd.to_numeric(raw["u10"], errors="coerce")
        v = pd.to_numeric(raw["v10"], errors="coerce")
        values.setdefault("WS", (u.pow(2) + v.pow(2)).pow(0.5))
        values.setdefault("WD", (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0)
        used.update({"u10", "v10"})
    if "GTI" not in values and "GHI" in values:
        values["GTI"] = values["GHI"]
    if "DNI" not in values and "GHI" in values:
        values["DNI"] = values["GHI"]
    if "DHI" not in values and "GHI" in values:
        values["DHI"] = values["GHI"]
    # Keep extra numeric columns as deterministic fallback channels.
    for name in numeric:
        if name not in used and name not in values:
            values[name] = pd.to_numeric(raw[name], errors="coerce")
    return pd.DataFrame(values)


def _interpolate_issue(frame: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate one issue onto the 15-minute Beijing-time grid."""
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    start, end = frame.index.min(), frame.index.max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise NwpInputError("NWP issue has fewer than two distinct valid times")
    target = pd.date_range(start, end, freq="15min")
    return frame.reindex(target).interpolate(method="time", limit_area="inside")


def load_nwp_issues(path: Path, kind: str) -> tuple[dict[pd.Timestamp, pd.DataFrame], list[str]]:
    """Load one station NWP CSV into per-issue 15-minute frames.

    Required columns are ``ISSUE_TIME`` (UTC) and ``DATE_TIME`` (valid time,
    UTC).  Additional columns are the physical NWP variables.  A long CSV may
    contain several issues; rows are grouped by issue.
    """
    if not path.exists():
        raise FileNotFoundError(f"NWP file does not exist: {path}")
    raw = pd.read_csv(path, low_memory=False)
    if "ISSUE_TIME" not in raw or "DATE_TIME" not in raw:
        raise NwpInputError(f"{path} must contain ISSUE_TIME and DATE_TIME (UTC)")
    issue = _utc(raw["ISSUE_TIME"])
    valid = _utc(raw["DATE_TIME"])
    if issue.isna().any() or valid.isna().any():
        raise NwpInputError(f"{path} contains unparseable ISSUE_TIME/DATE_TIME values")
    columns = [
        name for name in raw.columns
        if name not in NWP_META and pd.api.types.is_numeric_dtype(raw[name])
    ]
    canonical = _canonicalize(raw, columns)
    variables = _pick_variables(list(canonical.columns), kind)
    frame = canonical[variables].copy()
    frame.index = (valid + CN_OFFSET).dt.tz_localize(None)
    frame["__issue_cn"] = (issue + CN_OFFSET).dt.tz_localize(None).to_numpy()
    issues: dict[pd.Timestamp, pd.DataFrame] = {}
    for issue_cn, group in frame.groupby("__issue_cn", sort=True):
        values = group.drop(columns="__issue_cn")
        issues[pd.Timestamp(issue_cn)] = _interpolate_issue(values)
    if not issues:
        raise NwpInputError(f"{path} contains no usable NWP issue")
    return issues, variables


def find_station_nwp(nwp_dir: Path | str | None, station: str) -> Path:
    """Resolve the station-level NWP file used by ``prepare``."""
    if nwp_dir is None:
        raise NwpInputError("--nwp-dir is required when --future-source is 'nwp'")
    nwp_dir = Path(nwp_dir)
    candidates = [
        nwp_dir / f"zdata_{station}_nwp.csv",
        nwp_dir / f"zdata_{station}_nwp_15min.csv",
        nwp_dir / f"{station}_nwp.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No station NWP file for {station}; expected one of "
        + ", ".join(candidate.name for candidate in candidates)
    )


def _wind_components(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ws_name = next((name for name in ("WS", "WS_100", "WS_HUBHEIGHT") if name in frame), None)
    wd_name = next((name for name in ("WD", "WD_100", "WD_HUBHEIGHT") if name in frame), None)
    if ws_name is None or wd_name is None:
        raise NwpInputError("wind NWP must provide a wind-speed and wind-direction variable")
    ws = frame[ws_name].astype(float).to_numpy()
    wd = frame[wd_name].astype(float).to_numpy()
    return (
        pd.Series(ws * np.cos(np.deg2rad(wd)), index=frame.index),
        pd.Series(ws * np.sin(np.deg2rad(wd)), index=frame.index),
    )


def build_nwp_future(
    index: pd.DatetimeIndex,
    positions: np.ndarray,
    horizon: int,
    kind: str,
    nwp_path: Path,
    calendar: np.ndarray,
) -> tuple[np.ndarray, list[str], str]:
    """Build ``[windows, horizon, 18]`` future conditions for each origin.

    ``index`` is the Beijing-time 15-minute station index.  For every origin the
    latest issue that is not newer than the origin and that covers the whole
    horizon is used.  Windows without a causally valid issue are returned as
    NaN and later rejected by the sample builder.
    """
    issues, variables = load_nwp_issues(nwp_path, kind)
    issue_times = sorted(issues)
    feature_names = ["future__wind_u", "future__wind_v"] + [f"nwp__{name}" for name in variables]
    feature_names += ["future__" + name for name in ["sin_day", "cos_day", "sin_week", "cos_week", "sin_year", "cos_year"]]
    if len(feature_names) != 18:
        raise AssertionError("future-condition contract must be 18 features")
    windows = np.full((len(positions), horizon, 18), np.nan, dtype=np.float32)
    for row, position in enumerate(positions):
        origin = pd.Timestamp(index[position])
        target = pd.date_range(origin, periods=horizon, freq="15min")
        for issue_time in reversed(issue_times):
            if issue_time > origin:
                continue
            frame = issues[issue_time]
            if frame.index.min() > target[0] or frame.index.max() < target[-1]:
                continue
            window = frame.reindex(target)
            if window.isna().any().any():
                continue
            if kind == "wind":
                u, v = _wind_components(window)
            else:
                u = window.get("GHI", window.get("DNI", pd.Series(0.0, index=target)))
                v = window.get("GTI", window.get("DHI", pd.Series(0.0, index=target)))
            physical = window[variables].to_numpy(dtype=np.float32)
            values = np.column_stack(
                [u.to_numpy(), v.to_numpy(), physical, calendar[position:position + horizon]]
            ).astype(np.float32)
            windows[row] = values
            break
    return windows, feature_names, nwp_path.name
