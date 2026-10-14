from dataclasses import dataclass
from datetime import datetime, timezone
import re


_PATTERN = re.compile(r"^tower_(?P<layer>sfc|pl)_(?P<stamp>\d{8}_\d{2})_short\.grib2$")


@dataclass(frozen=True)
class NwpSelection:
    sfc: str
    pl: str
    issue_sfc_utc: datetime
    issue_pl_utc: datetime


def _parse(path: str):
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    match = _PATTERN.match(name)
    if not match:
        return None
    stamp = datetime.strptime(match.group("stamp"), "%Y%m%d_%H").replace(tzinfo=timezone.utc)
    return match.group("layer"), stamp, path


def select_latest_files(files: list[str], require_common_cycle: bool = False) -> NwpSelection:
    normalized_files: list[str] = []
    for raw in files:
        normalized_files.extend(raw.replace("\\n", "\n").splitlines())
    parsed = [item for path in normalized_files if (item := _parse(path.strip()))]
    grouped: dict[str, dict[datetime, str]] = {"sfc": {}, "pl": {}}
    for layer, stamp, path in parsed:
        grouped[layer][stamp] = path
    if not grouped["sfc"] or not grouped["pl"]:
        raise ValueError("both sfc and pl NWP files are required")
    if require_common_cycle:
        common = set(grouped["sfc"]) & set(grouped["pl"])
        if not common:
            raise ValueError("no common NWP issue cycle")
        stamp = max(common)
        return NwpSelection(grouped["sfc"][stamp], grouped["pl"][stamp], stamp, stamp)
    sfc_stamp = max(grouped["sfc"])
    pl_stamp = max(grouped["pl"])
    return NwpSelection(grouped["sfc"][sfc_stamp], grouped["pl"][pl_stamp], sfc_stamp, pl_stamp)
