import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from kerr_sbi.config import load_config
from kerr_sbi.persistence import exclusive_lock, write_json
from kerr_sbi.reference import RenderCache, prior_axes


def generate(directory: Path, cfg: dict, settings: dict) -> None:
    cache = RenderCache(directory / "renders", cfg)
    for size in settings["reference"]["grid_sizes"]:
        axes = prior_axes(cfg, size)
        images = np.empty((size, size, 64, 64), dtype=np.float32)
        for count, (i, j) in enumerate(np.ndindex(size, size), 1):
            inclination = float(np.rad2deg(np.arccos(axes[1][j])))
            inclination = float(
                np.clip(inclination, cfg["prior"]["incl_min_deg"], cfg["prior"]["incl_max_deg"])
            )
            images[i, j] = cache.image(float(axes[0][i]), inclination)
            write_json(
                directory / "status.json",
                {"stage": "render", "grid_size": size, "done": count, "total": size * size},
            )
            if count % 8 == 0 or count == size * size:
                print(f"Grid {size}: {count}/{size * size}", flush=True)
        temporary = directory / "grid.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, a=axes[0], c=axes[1], images=images)
        temporary.replace(directory / f"grid_{size}.npz")
    write_json(directory / "status.json", {"stage": "grids_complete"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="local_pilot_20260912")
    parser.add_argument("--settings", type=Path, default=Path("configs/reference.toml"))
    parser.add_argument("--stage", choices=("grid", "analyze", "disk-edge", "all"), default="all")
    args = parser.parse_args()
    if not args.run or not all(c.isascii() and (c.isalnum() or c in "-_") for c in args.run):
        raise ValueError("invalid run name")
    root = load_config()["project_root"]
    checkpoint = root / "runs" / args.checkpoint
    cfg = {**json.loads((checkpoint / "run.json").read_text())["config"], "project_root": root}
    settings = tomllib.loads(args.settings.read_text())
    directory = root / "runs" / args.run
    directory.mkdir(parents=True, exist_ok=True)
    data_directory = root / "data" / "reference" / args.run
    data_directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".lock"):
        identity = {"checkpoint": args.checkpoint, "settings": settings}
        path = directory / "experiment.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("experiment settings differ")
        write_json(path, identity)
        if args.stage in ("grid", "all"):
            generate(data_directory, cfg, settings)
        if args.stage in ("analyze", "all"):
            from kerr_sbi.reference_diagnostics import analyze

            analyze(directory, checkpoint, settings, grid_directory=data_directory)
        if args.stage in ("disk-edge", "all"):
            from kerr_sbi.disk_edge import disk_edge_check

            disk_edge_check(directory, cfg, settings, cache_directory=data_directory)


if __name__ == "__main__":
    main()
