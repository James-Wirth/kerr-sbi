import hashlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kerr_sbi.persistence import file_sha256

DEFAULT_AREAS = ("data", "runs", "results", "logs", ".tools")
UNIT_DEPTHS = {"data/datasets": 3, "data/reference": 3, "logs": 1, ".tools": 2}
SKIPPED_PARTS = frozenset({"__pycache__"})


def unit_of(relative: str) -> str:
    parts = PurePosixPath(relative).parts
    depth = 2
    for prefix, value in UNIT_DEPTHS.items():
        if parts[: len(PurePosixPath(prefix).parts)] == PurePosixPath(prefix).parts:
            depth = value
    return "/".join(parts[: min(depth, len(parts))])


def file_records(root: Path, areas: Iterable[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for area in areas:
        base = root / area
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if SKIPPED_PARTS.intersection(path.relative_to(root).parts):
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                records[relative] = {"symlink": str(path.readlink())}
            elif path.is_file():
                stat = path.stat()
                records[relative] = {
                    "bytes": stat.st_size,
                    "sha256": file_sha256(path),
                    "inode": [stat.st_dev, stat.st_ino],
                }
    return records


def summarize_units(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for relative in sorted(records):
        grouped.setdefault(unit_of(relative), []).append(relative)
    result = {}
    for unit, members in grouped.items():
        digest = hashlib.sha256()
        seen, disk = set(), 0
        for relative in members:
            record = records[relative]
            inner = PurePosixPath(relative).relative_to(unit).as_posix() if relative != unit else ""
            digest.update(f"{inner}\0{record.get('sha256', record.get('symlink'))}\n".encode())
            inode = tuple(record.get("inode", ()))
            if inode and inode not in seen:
                seen.add(inode)
                disk += record["bytes"]
        result[unit] = {
            "files": len(members),
            "bytes": sum(records[relative].get("bytes", 0) for relative in members),
            "disk_bytes": disk,
            "content_sha256": digest.hexdigest(),
        }
    return result


def disk_bytes(records: Mapping[str, Mapping[str, Any]]) -> int:
    seen: dict[tuple[int, ...], int] = {}
    for record in records.values():
        if "inode" in record:
            seen[tuple(record["inode"])] = record["bytes"]
    return sum(seen.values())


def snapshot(root: Path, areas: Iterable[str] = DEFAULT_AREAS) -> dict[str, Any]:
    areas = list(areas)
    records = file_records(root, areas)
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "areas": areas,
        "logical_bytes": sum(record.get("bytes", 0) for record in records.values()),
        "disk_bytes": disk_bytes(records),
        "units": summarize_units(records),
        "files": records,
    }


def relocate(relative: str, moves: Mapping[str, str]) -> str:
    matches = [
        source
        for source in moves
        if relative == source or relative.startswith(source.rstrip("/") + "/")
    ]
    if not matches:
        return relative
    source = max(matches, key=len)
    return moves[source] + relative[len(source) :]


def compare(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    moves: Mapping[str, str] | None = None,
    removed: Iterable[str] = (),
) -> dict[str, Any]:
    moves = moves or {}
    removed = [prefix.rstrip("/") for prefix in removed]
    old, new = before["files"], after["files"]
    matched, missing, changed, moved, deleted = set(), [], [], [], []
    for relative, record in old.items():
        if any(relative == prefix or relative.startswith(prefix + "/") for prefix in removed):
            if relative in new:
                changed.append(relative)
            deleted.append(relative)
            continue
        target = relocate(relative, moves)
        if target not in new:
            missing.append(relative)
            continue
        matched.add(target)
        if new[target].get("sha256", new[target].get("symlink")) != record.get(
            "sha256", record.get("symlink")
        ):
            changed.append(relative)
        elif target != relative:
            moved.append(relative)
    return {
        "compared_files": len(old),
        "unchanged_or_moved": len(old) - len(missing) - len(changed) - len(deleted),
        "moved": len(moved),
        "deleted_as_declared": len(deleted),
        "deleted_bytes": sum(old[relative].get("bytes", 0) for relative in deleted),
        "missing": sorted(missing),
        "changed": sorted(changed),
        "added": sorted(set(new) - matched),
        "disk_bytes_before": before["disk_bytes"],
        "disk_bytes_after": after["disk_bytes"],
    }


def catalog_coverage(
    catalog: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    entries = catalog["entries"]
    paths = [entry["path"].rstrip("/") for entry in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("catalog paths must be unique")

    def owner(unit: str) -> str | None:
        candidates = [path for path in paths if unit == path or unit.startswith(path + "/")]
        return max(candidates, key=len) if candidates else None

    owned: dict[str, list[str]] = {path: [] for path in paths}
    uncatalogued, hidden = [], []
    for unit in units:
        if PurePosixPath(unit).name.startswith("."):
            hidden.append(unit)
            continue
        path = owner(unit)
        if path is None:
            uncatalogued.append(unit)
        else:
            owned[path].append(unit)
    enriched = []
    for entry, path in zip(entries, paths, strict=True):
        members = owned[path]
        enriched.append(
            {
                **entry,
                "units": len(members),
                "files": sum(units[unit]["files"] for unit in members),
                "bytes": sum(units[unit]["bytes"] for unit in members),
                "disk_bytes": sum(units[unit]["disk_bytes"] for unit in members),
                "content_sha256": units[members[0]]["content_sha256"]
                if len(members) == 1 and members[0] == path
                else None,
            }
        )
    totals: dict[str, int] = {}
    for entry in enriched:
        totals[entry["class"]] = totals.get(entry["class"], 0) + entry["disk_bytes"]
    return {
        "entries": enriched,
        "uncatalogued_units": sorted(uncatalogued),
        "hidden_units": sorted(hidden),
        "empty_entries": sorted(path for path, members in owned.items() if not members),
        "disk_bytes_by_class": totals,
    }
