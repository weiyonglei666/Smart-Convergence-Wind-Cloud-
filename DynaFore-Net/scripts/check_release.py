#!/usr/bin/env python
"""Fail if a source tree contains datasets, weights, links or private paths."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

CONTACT = "ts26170007a31@cumt.edu.cn"
FORBIDDEN_SUFFIXES = {".csv", ".tsv", ".xls", ".xlsx", ".npy", ".npz", ".parquet", ".h5", ".hdf5",
                      ".nc", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".log", ".zip", ".tar", ".gz"}
GENERATED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".venv", "venv", "build", "dist"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".in"}


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in GENERATED_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink() or path.is_file():
            yield path


def audit(root: Path):
    errors = []
    files = list(source_files(root))
    for path in files:
        rel = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"Symlink is not permitted: {rel}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Excluded data/artifact file: {rel}")
        if any(part in {"data", "datasets", "raw", "outputs", "runs", "checkpoints"} for part in rel.parts):
            errors.append(f"Private data/artifact directory: {rel}")
        if path.suffix in TEXT_SUFFIXES or path.name == "LICENSE":
            text = path.read_text(encoding="utf-8")
            if re.search(r"/(?:mnt/workspace|home/ubuntu)(?:/|\b)", text):
                errors.append(f"Private absolute server path: {rel}")
            if re.search(r"(?m)^-{5}BEGIN (?:[A-Z]+ )?PRIVATE KEY-{5}$", text):
                errors.append(f"Private key marker: {rel}")
    for name in ["README.md", "README_EN.md", "DATA_AVAILABILITY.md"]:
        path = root / name
        if not path.is_file() or CONTACT not in path.read_text(encoding="utf-8"):
            errors.append(f"Missing dataset contact in {name}")
    return files, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    files, errors = audit(args.root.resolve())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(files)} source/documentation files; no datasets, checkpoints or private absolute paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
