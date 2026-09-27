import argparse
import json
from pathlib import Path

from kerr_sbi.inventory import DEFAULT_AREAS, catalog_coverage, compare, snapshot
from kerr_sbi.persistence import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash local artifacts, compare snapshots and check the artifact catalog"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    take = commands.add_parser("snapshot", help="record size and SHA-256 of every artifact file")
    take.add_argument("--out", type=Path, required=True)
    take.add_argument("--areas", nargs="+", default=list(DEFAULT_AREAS))
    diff = commands.add_parser("compare", help="verify files survived relocation unchanged")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--moves", type=Path, help="JSON object mapping old to new path prefixes")
    diff.add_argument("--removed", type=Path, help="JSON list of intentionally deleted prefixes")
    diff.add_argument("--out", type=Path)
    cover = commands.add_parser("catalog", help="attach sizes to a curated artifact catalog")
    cover.add_argument("--catalog", type=Path, required=True)
    cover.add_argument("--snapshot", type=Path, required=True)
    cover.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "snapshot":
        result = snapshot(root, args.areas)
        write_json(args.out, result)
        print(json.dumps({key: result[key] for key in ("logical_bytes", "disk_bytes")}))
        return
    if args.command == "compare":
        moves = json.loads(args.moves.read_text()) if args.moves else {}
        removed = json.loads(args.removed.read_text()) if args.removed else []
        result = compare(
            json.loads(args.before.read_text()), json.loads(args.after.read_text()), moves, removed
        )
        if args.out:
            write_json(args.out, result)
        print(json.dumps({**result, "added": len(result["added"])}, indent=2))
        if result["missing"] or result["changed"]:
            raise SystemExit(1)
        return
    units = json.loads(args.snapshot.read_text())["units"]
    result = catalog_coverage(json.loads(args.catalog.read_text()), units)
    write_json(args.out, result)
    print(
        json.dumps(
            {key: result[key] for key in ("uncatalogued_units", "empty_entries")}
            | {"disk_bytes_by_class": result["disk_bytes_by_class"]},
            indent=2,
        )
    )
    if result["uncatalogued_units"] or result["empty_entries"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
