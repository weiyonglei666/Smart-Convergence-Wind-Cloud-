#!/usr/bin/env python
"""Audit and zip an allowlisted source tree; never upload or publish anything."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from check_release import audit

ROOT_FILES = {"README.md", "README_EN.md", "LICENSE", "DATA_AVAILABILITY.md", "THIRD_PARTY_NOTICES.md",
              "CONTRIBUTING.md", "pyproject.toml", "requirements.txt", "MANIFEST.in", ".gitignore"}
ROOT_DIRS = {"src", "tests", "configs", "docs", "scripts"}
EXTENSIONS = {".py", ".md", ".json", ".toml", ".txt", ".in"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="ZIP outside the source directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    target = (args.destination or root.parent / "DynaFore-Net_source.zip").resolve()
    if target.is_relative_to(root):
        parser.error("ZIP destination must be outside the source tree")
    if target.exists() and not args.overwrite:
        parser.error(f"Destination already exists: {target}")
    files, errors = audit(root)
    if errors:
        parser.error("Release audit failed:\n" + "\n".join(errors))
    selected = []
    for path in files:
        rel = path.relative_to(root)
        if (len(rel.parts) == 1 and rel.name in ROOT_FILES) or (rel.parts[0] in ROOT_DIRS and path.suffix in EXTENSIONS):
            selected.append((path, rel))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w" if args.overwrite else "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, rel in selected:
            archive.write(path, Path("DynaFore-Net") / rel)
    print(f"Created {target.name}: {len(selected)} files, {target.stat().st_size} bytes; no data or weights")


if __name__ == "__main__":
    main()
