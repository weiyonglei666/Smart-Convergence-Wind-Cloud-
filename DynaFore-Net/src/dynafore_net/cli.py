"""Portable prepare/train/evaluate entry point with no machine-specific paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import discover_power_files, prepare_station
from .evaluation import evaluate_run
from .training import DYNAFORE_ABLATIONS, train_station
from .utils import write_json

DEFAULTS = {
    "lookback": 288, "pred_len": 16, "epochs": 20, "min_epochs": 6, "patience": 4,
    "scheduler_patience": 2, "batch_size": 256, "d_model": 32, "layers": 1,
    "lr": .001, "dropout": .1, "seed": 20261011, "threads": 4,
    "max_train": 0, "max_val": 0, "max_test": 0,
}
PRIMARY_METRICS = ["MSE@16", "MAE@16", "nMAE@16", "nRMSE@16", "ACC@16"]


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    defaults = DEFAULTS.copy()
    if known.config:
        config = json.loads(known.config.read_text(encoding="utf-8"))
        unknown = set(config) - set(defaults)
        if unknown:
            pre.error(f"Unknown config keys: {sorted(unknown)}")
        defaults.update(config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "run", "evaluate", "summarize"])
    parser.add_argument("--config", type=Path, help="Optional JSON training configuration")
    parser.add_argument("--raw-dir", type=Path, help="User-supplied private CSV directory")
    parser.add_argument("--capacity-json", type=Path, help="JSON mapping station IDs to capacity in MW")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--station", nargs="+", help="Station IDs; default: all discovered/prepared stations")
    parser.add_argument("--variants", nargs="+", choices=DYNAFORE_ABLATIONS, default=["full"])
    parser.add_argument("--device", default="cpu", help="cpu or an explicitly selected cuda:N")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing existing prepared data/runs")
    for key, default in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(DEFAULTS[key]), default=default)
    args = parser.parse_args(argv)
    if args.pred_len != 16:
        parser.error("This release uses exactly 16 forecast steps")
    if args.lookback < 32 or args.lookback % 16:
        parser.error("--lookback must be a multiple of 16 and at least 32")
    if min(args.epochs, args.min_epochs, args.patience, args.batch_size, args.threads, args.layers) < 1:
        parser.error("Epoch, patience, batch, thread and layer settings must be positive")
    if args.min_epochs > args.epochs:
        parser.error("--min-epochs must not exceed --epochs")
    if args.scheduler_patience < 0 or min(args.max_train, args.max_val, args.max_test) < 0:
        parser.error("Scheduler patience and window limits must be nonnegative")
    if args.d_model <= 0 or args.d_model % 4 or args.lr <= 0 or not 0 <= args.dropout < 1:
        parser.error("d_model must be divisible by 4; lr > 0; 0 <= dropout < 1")
    if args.action in ["prepare", "run"] and (args.raw_dir is None or not args.raw_dir.is_dir()):
        parser.error("prepare/run requires an existing --raw-dir")
    args.output = args.output.resolve()
    return args


def prepare_all(args):
    found = discover_power_files(args.raw_dir)
    wanted = set(args.station or found)
    missing = wanted - set(found)
    if not wanted or missing:
        raise ValueError(f"No matching raw power files; missing stations: {sorted(missing)}")
    manifest = args.output / "data" / "stations.json"
    if manifest.exists() and not args.overwrite:
        raise FileExistsError("Prepared data already exists; use train, or --overwrite to prepare again")
    metas = [prepare_station(station, kind, path, args.output, args)
             for station, (kind, path) in sorted(found.items()) if station in wanted]
    write_json(manifest, metas)
    write_json(args.output / "protocol.json", {
        "model": "DynaFore-Net", "lookback": args.lookback, "pred_len": 16, "sampling_minutes": 15,
        "split": "chronological 70/15/15 with target-overlap purge",
        "future_weather_source": "measured_future",
        "future_weather_note": "Task-permitted future measured tower/qxz observations, not causally available NWP",
        "missing_weather": "causal forward fill up to 4 steps; no backward fill",
        "capacity": "must be supplied; never estimated from test targets",
        "window_limits": {key: getattr(args, key) for key in ["max_train", "max_val", "max_test"]},
        "primary_metrics": PRIMARY_METRICS,
    })
    for meta in metas:
        print(json.dumps({"prepared": meta["station"], "counts": meta["counts"]}), flush=True)


def selected_stations(args):
    manifest = args.output / "data" / "stations.json"
    if not manifest.exists():
        raise FileNotFoundError("No prepared data; run prepare first")
    metas = json.loads(manifest.read_text(encoding="utf-8"))
    available = {meta["station"] for meta in metas}
    wanted = set(args.station or available)
    if not wanted or wanted - available:
        raise ValueError(f"Unprepared station IDs: {sorted(wanted - available)}")
    return sorted(wanted)


def train_all(args):
    stations = selected_stations(args)
    for station in stations:
        for variant in args.variants:
            run = args.output / "runs" / station / variant.replace(" ", "_")
            if (run / "checkpoint.pt").exists() and not args.overwrite:
                raise FileExistsError(f"Run already exists: {run}; choose another output or --overwrite")
            write_json(run / "training_config.json", {**{key: getattr(args, key) for key in DEFAULTS},
                                                     "station": station, "variant": variant, "device": args.device})
            train_station(station, args.output / "data", args, run, args.device, variant)


def summarize(out):
    rows = []
    for path in sorted((out / "runs").glob("*/*/metrics_test.json")):
        station = path.parent.parent.name
        meta = json.loads((out / "data" / station / "meta.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"station": station, "kind": meta["kind"], "capacity_mw": meta["capacity_mw"], **result})
    if not rows:
        raise ValueError("No completed test results")
    table = pd.DataFrame(rows)
    table.to_csv(out / "results_by_station.csv", index=False)
    table.groupby("variant")[PRIMARY_METRICS].mean().to_csv(out / "results_variant_mean.csv")
    table.groupby(["kind", "variant"])[PRIMARY_METRICS].mean().to_csv(out / "results_kind_variant_mean.csv")
    # Count per-station best scores; all tied methods receive one count.
    counts = {variant: {metric: 0 for metric in PRIMARY_METRICS} for variant in sorted(table.variant.unique())}
    for _, group in table.groupby("station"):
        for metric in PRIMARY_METRICS:
            best = group[metric].max() if metric == "ACC@16" else group[metric].min()
            for variant in group.loc[np.isclose(group[metric], best, rtol=1e-10, atol=1e-10), "variant"]:
                counts[variant][metric] += 1
    pd.DataFrame.from_dict(counts, orient="index").rename_axis("variant").to_csv(out / "1st_count.csv")
    print(table[["station", "variant"] + PRIMARY_METRICS].to_string(index=False))


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.action in ["prepare", "run"]:
        prepare_all(args)
    if args.action in ["train", "run"]:
        train_all(args)
        summarize(args.output)
    elif args.action == "evaluate":
        for station in selected_stations(args):
            for variant in args.variants:
                run = args.output / "runs" / station / variant.replace(" ", "_")
                evaluate_run(args.output / "data", run, args.device, args.batch_size)
        summarize(args.output)
    elif args.action == "summarize":
        summarize(args.output)
