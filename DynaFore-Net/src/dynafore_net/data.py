"""CSV loading and chronological window preparation; no data is bundled."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .nwp import build_nwp_future, find_station_nwp
from .utils import write_json

WEATHER_META = {"OBJ_ID", "DATE_TIME", "CREATE_TIME", "FARM_ID"}

# The report fixes the input contract at 17 historical and 18 future features.
HISTORY_FEATURES = 17
FUTURE_FEATURES = 18
WEATHER_FEATURES = 10

def discover_power_files(raw_dir: Path) -> dict[str, tuple[str, Path]]:
    found: dict[str, tuple[str, Path]] = {}
    for path in sorted(raw_dir.glob("zdata_*_power_real_15min.csv")):
        match = re.match(r"zdata_(.+)_(wind|pv)_power_real_15min\.csv$", path.name)
        if match:
            found[match.group(1)] = (match.group(2), path)
    return found


def weather_path(station: str, kind: str, raw_dir: Path) -> Path | None:
    suffix = "wind_tower" if kind == "wind" else "qxz"
    direct = raw_dir / f"zdata_{station}_{suffix}_15min.csv"
    if direct.exists():
        return direct
    history = raw_dir / f"zdata_{station}_{suffix}_15min_history.csv"
    return history if history.exists() else None


def capacity_map(raw_dir: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    wind = raw_dir / "zbase_wind_farm_info_encryption.csv"
    if wind.exists():
        df = pd.read_csv(wind)
        for _, row in df.iterrows():
            value = pd.to_numeric(row.get("CAP"), errors="coerce")
            if pd.notna(value) and float(value) > 0:
                result[str(row["FARM_ID"])] = float(value)
    pv = raw_dir / "zbase_photovoltaic_farm_info.csv"
    if pv.exists():
        df = pd.read_csv(pv)
        for _, row in df.iterrows():
            for col in ["CAP_REPORT", "CAP_AC", "CAP_DC"]:
                value = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(value) and float(value) > 0:
                    result[str(row["FARM_ID"])] = float(value)
                    break
    return result


def read_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "DATE_TIME" not in df:
        raise ValueError(f"缺少时间列: {path}")
    dates = df["DATE_TIME"].astype(str)
    iso = dates.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)")
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    dt.loc[iso] = pd.to_datetime(dates.loc[iso], format="ISO8601", errors="coerce")
    dt.loc[~iso] = pd.to_datetime(dates.loc[~iso], dayfirst=True, errors="coerce")
    df = df.loc[dt.notna()].copy()
    df["DATE_TIME"] = dt.loc[dt.notna()].to_numpy()
    df = df.set_index("DATE_TIME")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.groupby(level=0).mean(numeric_only=True)
    numeric.index = pd.DatetimeIndex(numeric.index).floor("15min")
    return numeric.groupby(level=0).mean(numeric_only=True).sort_index()


def choose_weather_columns(weather: pd.DataFrame, kind: str) -> list[str]:
    cols = [c for c in weather.columns if c not in WEATHER_META and pd.api.types.is_numeric_dtype(weather[c])]
    if kind == "pv":
        preferred = ["GHI", "GTI", "DNI", "DHI", "WS", "WD", "T", "H", "P", "T_BACKBOARD"]
    else:
        preferred = [
            "WS_HUBHEIGHT", "WS_100", "WS_140", "WS_150", "WS_10", "AVG_WS_100", "AVG_WS_140",
            "WD_HUBHEIGHT", "WD_100", "WD_140", "WD_150", "WD_10", "AVG_WD_100", "AVG_WD_140",
            "T", "H", "P", "T_10", "H_10", "P_10",
        ]
    chosen = [c for c in preferred if c in cols]
    # Data suppliers use different tower heights and prefixes.  Keep a small
    # deterministic fallback set, avoiding EXMAX/EXMIN/SD aggregates.
    fallback = [c for c in sorted(cols) if not c.startswith(("EXMAX_", "EXMIN_", "SD_", "AVG_"))]
    for c in fallback:
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= WEATHER_FEATURES:
            break
    if len(chosen) < WEATHER_FEATURES:
        raise ValueError(
            f"{kind} weather table exposes {len(chosen)} usable variables; "
            f"the documented model contract requires {WEATHER_FEATURES}"
        )
    if kind == "wind" and not (any(c.startswith("WS") for c in chosen) and any(c.startswith("WD") for c in chosen)):
        raise ValueError("wind weather table must provide wind speed (WS*) and direction (WD*)")
    if kind == "pv" and not (("GHI" in chosen) and ({"GTI", "DNI"} & set(chosen))):
        raise ValueError("PV weather table must provide GHI and at least one of GTI/DNI")
    # Keep the physically required channels even if a provider exposes more
    # than ten WS* columns before the first WD* column.
    if kind == "wind":
        required = [next(c for c in chosen if c.startswith("WS")), next(c for c in chosen if c.startswith("WD"))]
    else:
        required = ["GHI", next(c for c in chosen if c in {"GTI", "DNI"})]
    ordered = required + [c for c in chosen if c not in required]
    return ordered[:WEATHER_FEATURES]


def cyclical_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    minutes = index.hour * 60 + index.minute
    day = minutes / 1440.0
    year = (index.dayofyear.to_numpy() - 1 + day) / 365.25
    week = (index.dayofweek.to_numpy() + day) / 7.0
    return pd.DataFrame(
        {
            "sin_day": np.sin(2 * np.pi * day),
            "cos_day": np.cos(2 * np.pi * day),
            "sin_week": np.sin(2 * np.pi * week),
            "cos_week": np.cos(2 * np.pi * week),
            "sin_year": np.sin(2 * np.pi * year),
            "cos_year": np.cos(2 * np.pi * year),
        },
        index=index,
    )


def load_station_raw(station: str, kind: str, power_path: Path, raw_dir: Path, capacities: dict | None = None) -> tuple[pd.DataFrame, list[str], float, Path | None]:
    weather_file = weather_path(station, kind, raw_dir)
    if weather_file is None:
        raise FileNotFoundError(f"Missing measured weather CSV for station {station}")
    power = read_timeseries(power_path)
    if "ACTIVE_POWER" not in power:
        raise ValueError(f"功率文件没有 ACTIVE_POWER: {power_path}")
    weather = read_timeseries(weather_file) if weather_file else pd.DataFrame(index=power.index)
    start = min(power.index.min(), weather.index.min()) if len(weather) else power.index.min()
    end = max(power.index.max(), weather.index.max()) if len(weather) else power.index.max()
    index = pd.date_range(start.floor("15min"), end.floor("15min"), freq="15min")
    power = power.reindex(index)
    weather = weather.reindex(index)
    wcols = choose_weather_columns(weather, kind)
    # Historical inputs must never be reconstructed from later observations.
    # Forward fill at most four steps; remaining gaps invalidate the window.
    weather[wcols] = weather[wcols].ffill(limit=4)
    cal = cyclical_frame(index)
    base = pd.concat([power[["ACTIVE_POWER"]], weather[wcols], cal], axis=1)
    base.columns = ["power"] + [f"weather__{c}" for c in wcols] + list(cal.columns)
    base = base.replace([np.inf, -np.inf], np.nan)
    cap = (capacities or capacity_map(raw_dir)).get(station)
    if cap is None or not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"Missing positive capacity_mw for {station}; supply capacity metadata or --capacity-json")
    return base, wcols, float(cap), weather_file


def future_weather_frame(base: pd.DataFrame, weather_cols: list[str], kind: str) -> tuple[np.ndarray, list[str]]:
    raw = base[[f"weather__{c}" for c in weather_cols]].to_numpy(dtype=np.float32)
    names = [f"weather__{c}" for c in weather_cols]
    # The measured-future ablation reuses observed tower/qxz values.  Wind
    # components are derived from the measured speed/direction; this is *not*
    # NWP and is reported separately in metadata.
    ws_col = next((f"weather__{c}" for c in weather_cols if c.startswith("WS")), None)
    wd_col = next((f"weather__{c}" for c in weather_cols if c.startswith("WD")), None)
    if kind == "wind" and ws_col:
        ws = base[ws_col].to_numpy(dtype=np.float32)
        wd = base[wd_col].to_numpy(dtype=np.float32) if wd_col else np.zeros_like(ws)
        u, v = ws * np.cos(np.deg2rad(wd)), ws * np.sin(np.deg2rad(wd))
    elif kind == "pv":
        first = base.get("weather__GHI", pd.Series(0.0, index=base.index)).to_numpy(dtype=np.float32)
        second = base.get("weather__GTI", base.get("weather__DNI", pd.Series(0.0, index=base.index))).to_numpy(dtype=np.float32)
        u, v = first, second
    else:
        u = v = np.zeros(len(base), dtype=np.float32)
    cal = base[["sin_day", "cos_day", "sin_week", "cos_week", "sin_year", "cos_year"]].to_numpy(dtype=np.float32)
    fut = np.column_stack([u, v, raw, cal]).astype(np.float32)
    fut_names = ["future__wind_u", "future__wind_v"] + names + [f"future__{c}" for c in ["sin_day", "cos_day", "sin_week", "cos_week", "sin_year", "cos_year"]]
    if fut.shape[-1] != FUTURE_FEATURES:
        raise AssertionError(f"future-condition contract must be {FUTURE_FEATURES} features")
    return fut, fut_names


def _subsample(index: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(index) <= limit:
        return index
    return index[np.linspace(0, len(index) - 1, limit).round().astype(int)]


def prepare_station(station: str, kind: str, power_path: Path, out: Path, args: argparse.Namespace) -> dict:
    capacities = json.loads(args.capacity_json.read_text(encoding="utf-8")) if args.capacity_json else None
    base, wcols, cap, weather_file = load_station_raw(station, kind, power_path, args.raw_dir, capacities)
    xraw = base.to_numpy(dtype=np.float32)
    if xraw.shape[1] != HISTORY_FEATURES:
        raise AssertionError(
            f"history-condition contract must be {HISTORY_FEATURES} features, got {xraw.shape[1]}"
        )
    yraw = xraw[:, 0].copy()
    future_source = getattr(args, "future_source", "nwp")
    if future_source not in {"nwp", "measured"}:
        raise ValueError("future_source must be 'nwp' or 'measured'")
    lookback = int(getattr(args, "lookback_pv", 96) if kind == "pv" else args.lookback)
    L, H = lookback, args.pred_len
    positions = np.arange(L, len(base) - H + 1, dtype=np.int64)
    if future_source == "nwp":
        nwp_path = find_station_nwp(getattr(args, "nwp_dir", None), station)
        future, fnames, future_file = build_nwp_future(
            base.index, positions, H, kind, nwp_path,
            base[["sin_day", "cos_day", "sin_week", "cos_week", "sin_year", "cos_year"]].to_numpy(dtype=np.float32),
        )
    else:
        fraw_full, fnames = future_weather_frame(base, wcols, kind)
        future = np.stack([fraw_full[p:p + H] for p in positions]).astype(np.float32)
        future_file = str(weather_file) if weather_file else None
    if future.shape[1:] != (H, FUTURE_FEATURES):
        raise AssertionError(
            f"future-condition contract must be ({H}, {FUTURE_FEATURES}), got {future.shape[1:]}"
        )
    valid_x = np.isfinite(xraw).all(axis=1)
    valid_y = np.isfinite(yraw)
    valid_f = np.isfinite(future).all(axis=(1, 2))
    bad_x = (~valid_x).astype(np.int64)
    bad_y = (~valid_y).astype(np.int64)
    cx, cy = np.r_[0, np.cumsum(bad_x)], np.r_[0, np.cumsum(bad_y)]
    good = (cx[positions] - cx[positions - L] == 0) & (cy[positions + H] - cy[positions] == 0) & valid_f
    positions = positions[good]
    if len(positions) < 100:
        raise ValueError(f"{station} 有效窗口不足: {len(positions)}")
    x = np.stack([xraw[p - L:p] for p in positions]).astype(np.float32)
    f = future[good].astype(np.float32)
    y = np.stack([yraw[p:p + H] for p in positions]).astype(np.float32)
    origins = pd.DatetimeIndex(base.index[positions], name="origin_time")
    n = len(origins)
    split = np.full(n, "test", dtype="U5")
    split[: int(n * 0.70)] = "train"
    split[int(n * 0.70): int(n * 0.85)] = "val"
    # Purge windows whose labels overlap the next split boundary.
    for current, nxt in [("train", "val"), ("val", "test")]:
        next_start = origins[split == nxt].min()
        target_end = origins + pd.Timedelta(minutes=15 * (H - 1))
        split[(split == current) & (target_end >= next_start)] = "purged"
    keep = np.concatenate([_subsample(np.flatnonzero(split == s), lim) for s, lim in [("train", args.max_train), ("val", args.max_val), ("test", args.max_test)]])
    keep.sort()
    x, f, y, origins, split = x[keep], f[keep], y[keep], origins[keep], split[keep]
    station_dir = out / "data" / station
    station_dir.mkdir(parents=True, exist_ok=True)
    np.save(station_dir / "x.npy", x)
    np.save(station_dir / "future_weather.npy", f)
    np.save(station_dir / "y.npy", y)
    np.save(station_dir / "origins.npy", origins.astype(str).to_numpy(dtype="U32"))
    np.save(station_dir / "split.npy", split)
    meta = {
        "station": station, "kind": kind, "capacity_mw": cap, "lookback": L, "pred_len": H,
        "weather_file": str(weather_file) if weather_file else None,
        "future_weather_file": future_file,
        "weather_columns": wcols, "x_columns": list(base.columns),
        "future_columns": fnames, "n_total_before_subsample": int(n),
        "counts": {s: int((split == s).sum()) for s in ["train", "val", "test", "purged"]},
        "future_weather_source": "nwp" if future_source == "nwp" else "measured_future",
        "future_weather_policy": (
            "ECMWF-style NWP with UTC+8 alignment, issue_time <= origin and 15-minute interpolation"
            if future_source == "nwp" else
            "future measured tower/qxz weather; an explicit ablation, not operational NWP"
        ),
        "raw_power_file": str(power_path),
    }
    write_json(station_dir / "meta.json", meta)
    return meta


def load_station_arrays(data_dir: Path, station: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    d = data_dir / station
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    # Do not return file-backed memmaps.  They keep *.npy handles open on
    # Windows, so a completed run cannot be moved or deleted and repeated
    # evaluation sees stale files.
    return (
        np.load(d / "x.npy"), np.load(d / "future_weather.npy"), np.load(d / "y.npy"),
        np.load(d / "origins.npy"), np.load(d / "split.npy"), meta,
    )


def standardize(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(x[train_idx].mean(axis=(0, 1), dtype=np.float64), dtype=np.float32)
    std = np.asarray(x[train_idx].std(axis=(0, 1), dtype=np.float64), dtype=np.float32)
    std = np.maximum(std, 1e-5)
    return ((x - mean) / std).astype(np.float32), mean, std


def split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.flatnonzero(split == name) for name in ["train", "val", "test"]}
