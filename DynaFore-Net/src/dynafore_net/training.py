"""Full-window station-wise training with AutoDA, early stopping and LR scheduling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .augmentation import FeatureAwareAugment
from .model import DynaForeNet
from .data import load_station_arrays, standardize, split_indices
from .losses import LossCoefficients, composite_loss
from .metrics import metrics, real_time_acc
from .utils import write_json

DYNAFORE_ABLATIONS = ["full", "noPMD-PVA-TRA", "noNSAN", "noAutoDA", "noguided correction"]

def dynafore_flags(variant: str, kind: str) -> dict[str, object]:
    """Return isolated architecture switches for the requested DynaFore run."""
    if variant not in DYNAFORE_ABLATIONS:
        raise ValueError(f"unknown DynaFore variant: {variant}")
    flags: dict[str, object] = {
        "energy_type": kind,
        "pmd": True, "pva": True, "tra": True, "nsan": True,
        "guided_correction": True,
    }
    if variant == "noPMD-PVA-TRA":
        flags.update({"pmd": False, "pva": False, "tra": False})
    elif variant == "noNSAN":
        flags["nsan"] = False
    elif variant == "noguided correction":
        flags["guided_correction"] = False
    return flags


def apply_training_augmentation(xb, yb, augmenter, variant: str):
    """Apply document-specified AutoDA/fine reconstruction only when enabled."""
    if augmenter is None or variant == "noAutoDA":
        return xb, yb, xb.new_zeros(())
    x_aug, y_aug, regularizer = augmenter(xb, yb)
    return x_aug, y_aug, regularizer


def save_run(run: Path, model: str, station: str, cap: float, y: np.ndarray, pred: np.ndarray, origins: np.ndarray, split_name: str, extra: dict | None = None) -> dict:
    run.mkdir(parents=True, exist_ok=True)
    result = metrics(y, pred, cap)
    if extra:
        result.update(extra)
    write_json(run / f"metrics_{split_name}.json", result)
    np.save(run / f"prediction_{split_name}.npy", pred.astype(np.float32))
    pd.DataFrame({"origin_time": origins, "ACC16": real_time_acc(y, pred, cap, 16), "MAE16": np.abs(y - pred).mean(1)}).to_csv(run / f"per_origin_{split_name}.csv", index=False)
    return result


def train_station(station: str, data_dir: Path, args: argparse.Namespace, run: Path, device: str, variant: str = "full") -> None:
    x, fraw, y, origins, split, meta = load_station_arrays(data_dir, station)
    idx = split_indices(split)
    if any(len(j) == 0 for j in idx.values()):
        raise ValueError("Every train/validation/test split must contain valid windows")
    cap, kind, model_name = float(meta["capacity_mw"]), meta["kind"], "DynaFore-Net"
    run.mkdir(parents=True, exist_ok=True)
    import random
    import torch
    seed = args.seed + int.from_bytes(hashlib.sha256(station.encode()).digest()[:4], "big") % 100000
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.threads)
    xn, xm, xs = standardize(x, idx["train"])
    fn, fm, fs = standardize(fraw, idx["train"])
    future_columns = meta["future_columns"]
    yn = (y / cap).astype(np.float32)
    stats = {"x_mean": xm, "x_std": xs, "f_mean": fm, "f_std": fs}
    cfg = {"d_model": args.d_model, "layers": args.layers, "lr": args.lr, "dropout": args.dropout,
           "batch_size": args.batch_size, "future_columns": future_columns, "capacity_mw": cap,
           "n_heads": getattr(args, "n_heads", 4), "d_ff": getattr(args, "d_ff", 128),
           "nsan_window": getattr(args, "nsan_window", 72)}
    coefficients = LossCoefficients(
        step_decay=float(getattr(args, "loss_step_decay", 0.02)),
        ramp=float(getattr(args, "loss_ramp", 0.05)),
        consistency=float(getattr(args, "loss_consistency", 0.10)),
        frequency=float(getattr(args, "loss_frequency", 0.01)),
        statistics=float(getattr(args, "loss_statistics", 0.01)),
        mechanism=float(getattr(args, "loss_mechanism", 0.05)),
    )
    flags = dynafore_flags(variant, kind)
    torch_device = torch.device(device)
    model = DynaForeNet(xn.shape[-1], fn.shape[-1], xn.shape[1], yn.shape[1], stats, cfg, flags).to(torch_device)
    params = sum(p.numel() for p in model.parameters())
    augmenter = None
    if model_name == "DynaFore-Net" and variant != "noAutoDA":
        augmenter = FeatureAwareAugment(xn.shape[-1], stats, fine=True, power_scale=cap).to(torch_device)
    train_parameters = list(model.parameters()) + (list(augmenter.parameters()) if augmenter is not None else [])
    opt = torch.optim.AdamW(train_parameters, lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.scheduler_patience, min_lr=1e-5)
    train_idx = idx["train"].copy()
    best_score, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    best_augment = None
    def predict(indices: np.ndarray) -> np.ndarray:
        model.eval(); out = []
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                j = indices[start:start + args.batch_size]
                xb = torch.as_tensor(xn[j], device=torch_device)
                fb = torch.as_tensor(fn[j], device=torch_device)
                pred = model(xb, fb).float().detach().cpu().numpy() * cap
                out.append(pred)
        return np.concatenate(out, axis=0) if out else np.empty((0, y.shape[1]), dtype=np.float32)
    log = []
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(train_idx); model.train();
        if augmenter is not None:
            augmenter.train()
        losses = []
        for start in range(0, len(train_idx), args.batch_size):
            j = train_idx[start:start + args.batch_size]
            xb = torch.as_tensor(xn[j], device=torch_device)
            fb = torch.as_tensor(fn[j], device=torch_device)
            yb = torch.as_tensor(yn[j], device=torch_device)
            xb_train, yb_train, aug_regularizer = apply_training_augmentation(xb, yb, augmenter, variant)
            opt.zero_grad(set_to_none=True)
            clean_pred, aux = model(xb, fb, return_aux=True)
            augmented_pred = None
            if xb_train is not xb:
                augmented_pred, _ = model(xb_train, fb, return_aux=True)
            loss, parts = composite_loss(
                clean_pred, yb, augmented=augmented_pred, augmented_target=yb_train,
                aux=aux, energy_type=kind, coefficients=coefficients,
            )
            loss = loss + aug_regularizer
            parts["total"] = float(loss.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss epoch={epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(train_parameters, 1.0); opt.step()
            losses.append(parts)
        val_pred = predict(idx["val"])
        val_score = float(real_time_acc(y[idx["val"]], val_pred, cap, 16).mean())
        scheduler.step(val_score)
        row = {"epoch": epoch, "train_loss": float(np.mean([item["total"] for item in losses])),
               "val_ACC16": val_score, "lr": float(opt.param_groups[0]["lr"])}
        for name in ["task", "official", "ramp", "consistency", "frequency", "statistics", "mechanism"]:
            row[f"loss_{name}"] = float(np.mean([item[name] for item in losses]))
        log.append(row); pd.DataFrame(log).to_csv(run / "training_log.csv", index=False)
        print(json.dumps({"station": meta["station"], "model": model_name, **row}), flush=True)
        if val_score > best_score + 1e-5:
            best_score, best_epoch, stale = val_score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_augment = {k: v.detach().cpu().clone() for k, v in augmenter.state_dict().items()} if augmenter is not None else None
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred, test_pred = predict(idx["val"]), predict(idx["test"])
    extra = {"future_weather_used": True, "future_weather_source": meta["future_weather_source"], "parameters": params, "seed": seed, "epochs_trained": len(log), "best_epoch": best_epoch, "variant": variant, "auto_da_used": augmenter is not None, "lr_final": float(opt.param_groups[0]["lr"])}
    save_run(run, model_name, meta["station"], cap, y[idx["val"]], val_pred, origins[idx["val"]], "val", extra)
    save_run(run, model_name, meta["station"], cap, y[idx["test"]], test_pred, origins[idx["test"]], "test", extra)
    torch.save({"format_version": 1, "model": best_state or model.state_dict(),
                "stats": {key: value.tolist() for key, value in stats.items()},
                "capacity_mw": cap, "config": cfg, "variant": variant, "seed": seed,
                "best_epoch": best_epoch, "epochs_trained": len(log), "parameters": params,
                "augmentation_parameters": sum(p.numel() for p in augmenter.parameters()) if augmenter is not None else 0,
                "augmenter": best_augment, "auto_da_used": augmenter is not None, "meta": meta}, run / "checkpoint.pt")
    write_json(run / "complete.json", {"station": station, "variant": variant, "best_epoch": best_epoch,
                                     "epochs_trained": len(log), "future_weather_source": meta["future_weather_source"]})
