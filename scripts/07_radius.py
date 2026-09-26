import argparse
import json
import tomllib
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kerr_sbi.config import load_config
from kerr_sbi.obs_model import add_noise, preprocess_pfm
from kerr_sbi.persistence import exclusive_lock, file_sha256, write_json
from kerr_sbi.radius import RadiusCache, physical_radius
from kerr_sbi.reference import RenderCache, prior_axes


def render_points(
    cache: RadiusCache, points: np.ndarray, radius_max: float, label: str
) -> np.ndarray:
    radii = physical_radius(points, radius_max)
    images = []
    for n, (point, radius) in enumerate(zip(points, radii, strict=True), 1):
        inclination = float(np.rad2deg(np.arccos(point[1])))
        inclination = float(
            np.clip(
                inclination, cache.cfg["prior"]["incl_min_deg"], cache.cfg["prior"]["incl_max_deg"]
            )
        )
        images.append(cache.image(float(point[0]), inclination, float(radius)))
        if n % 8 == 0 or n == len(points):
            print(f"{label}: {n}/{len(points)}", flush=True)
    return np.stack(images)


def merge_nodes(
    points: list[np.ndarray], images: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    all_points, all_images = np.concatenate(points), np.concatenate(images)
    _, selected = np.unique(all_points, axis=0, return_index=True)
    selected.sort()
    return all_points[selected], all_images[selected]


def prepare(root: Path, name: str, settings: dict) -> tuple[Path, Path, dict, dict, RadiusCache]:
    options = settings["radius"]
    checkpoint = root / "runs" / options["checkpoint"]
    run = json.loads((checkpoint / "run.json").read_text())
    cfg = {**run["config"], "project_root": root}
    if run["effective_sigma_n"] != cfg["observation"]["sigma_n"]:
        raise ValueError("checkpoint and configured noise differ")
    if not 6 < options["radius_max"] < cfg["scene"]["r_out"]:
        raise ValueError("radius_max must be between the largest ISCO and the outer radius")
    source = root / "runs" / options["source_run"]
    summary = json.loads((source / "summary.json").read_text())
    if not summary["reference_certified"]:
        raise ValueError("the two-parameter reference must pass its numerical checks first")
    source_data = root / "data/reference" / options["source_run"]
    if file_sha256(source_data / "mesh.npz") != summary["mesh_sha256"]:
        raise ValueError("source mesh hash differs")
    directory = root / "runs" / name
    data_directory = root / "data/reference" / name
    directory.mkdir(parents=True, exist_ok=True)
    cache = RadiusCache(data_directory / "renders", cfg)
    if cache.identity != json.loads((source_data / "renders/identity.json").read_text()):
        raise ValueError("source mesh and new renders have different forward-model provenance")
    identity = {
        "settings": settings,
        "source_mesh_sha256": summary["mesh_sha256"],
        "source_summary_sha256": file_sha256(source / "summary.json"),
        "render_identity": cache.identity,
        "sigma_n": cfg["observation"]["sigma_n"],
    }
    identity_path = directory / "experiment.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("radius experiment identity differs")
    write_json(identity_path, identity)
    return directory, data_directory, cfg, options, cache


def generate(
    root: Path, directory: Path, data_directory: Path, cfg: dict, options: dict, cache: RadiusCache
) -> None:
    truth = np.array(
        [[a, np.cos(np.deg2rad(i)), t] for a, i in options["pairs"] for t in (0.0, 1.0)]
    )
    truth_images = render_points(cache, truth, options["radius_max"], "Truth images")
    baseline_cache = RenderCache(data_directory / "isco_checks", cfg)
    clamp_error = []
    for row, (a, inclination) in enumerate(options["pairs"]):
        baseline = baseline_cache.image(a, inclination)
        clamp_error.append(
            float(np.linalg.norm(baseline - truth_images[2 * row]) / cfg["observation"]["sigma_n"])
        )
    if max(clamp_error) > 1e-6:
        raise ValueError("explicit ISCO renders disagree with the simulator clamp")
    observations, cases = [], []
    radii = physical_radius(truth, options["radius_max"])
    for seed in options["noise_seeds"]:
        for row, (point, radius) in enumerate(zip(truth, radii, strict=True)):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), row // 2)
            observations.append(
                np.asarray(
                    add_noise(jnp.asarray(truth_images[row]), key, cfg["observation"]["sigma_n"])
                )
            )
            cases.append(
                {
                    "truth_index": row,
                    "spin": float(point[0]),
                    "inclination_deg": float(np.rad2deg(np.arccos(point[1]))),
                    "radius": float(radius),
                    "fraction": float(point[2]),
                    "noise_seed": seed,
                }
            )
    np.savez(
        data_directory / "observations.npz",
        truth=truth,
        images=truth_images,
        observations=np.stack(observations),
    )
    write_json(
        directory / "cases.json",
        {
            "cases": cases,
            "clamp_noise_norm": clamp_error,
            "noise_design": (
                "Paired noise within each spin/inclination pair and seed, across disk edges"
            ),
        },
    )
    source_data = root / "data/reference" / options["source_run"]
    with np.load(source_data / "mesh.npz") as source:
        base_points = np.column_stack([source["points"], np.zeros(len(source["points"]))])
        base_images = source["images"]
        bounds = np.column_stack([source["bounds"], [0.0, 1.0]])
    fixed_directory = root / "data/reference/reference_20260926/fixed_radius_renders"
    fixed_cfg = deepcopy(cfg)
    fixed_cfg["scene"]["r_in"] = options["radius_max"]
    fixed_identity = json.loads((fixed_directory / "identity.json").read_text())
    expected_identity = deepcopy(cache.identity)
    expected_identity["config"]["scene"]["r_in"] = options["radius_max"]
    if fixed_identity != expected_identity:
        raise ValueError("cached fixed-radius images have incompatible provenance")
    fixed_points, fixed_images = [], []
    for path in sorted(fixed_directory.glob("*.json")):
        if path.name == "identity.json":
            continue
        record = json.loads(path.read_text())
        if file_sha256(path.with_suffix(".pfm")) != record["sha256"]:
            raise ValueError("cached fixed-radius image is corrupt")
        a, inclination = record["parameters"]
        fixed_points.append([a, np.cos(np.deg2rad(inclination)), 1.0])
        fixed_images.append(preprocess_pfm(path.with_suffix(".pfm"), cfg))
    for size, fractions in zip(options["grid_sizes"], options["radius_fractions"], strict=True):
        axes = (*prior_axes(cfg, size), np.array(fractions))
        points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        images = render_points(cache, points, options["radius_max"], f"Grid {size}")
        points, images = merge_nodes(
            [base_points, np.array(fixed_points), points],
            [base_images, np.array(fixed_images), images],
        )
        np.savez(data_directory / f"grid_{size}.npz", points=points, images=images, bounds=bounds)
        write_json(directory / "status.json", {"stage": "grid", "size": size, "nodes": len(points)})
    write_json(directory / "status.json", {"stage": "grids_complete"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--settings", type=Path, default=Path("configs/radius.toml"))
    parser.add_argument("--stage", choices=("grid", "analyze", "all"), default="all")
    args = parser.parse_args()
    if not args.run or not all(c.isascii() and (c.isalnum() or c in "-_") for c in args.run):
        raise ValueError("invalid run name")
    root = load_config()["project_root"]
    settings = tomllib.loads(args.settings.read_text())
    with exclusive_lock(root / "runs" / args.run / ".lock"):
        directory, data_directory, cfg, options, cache = prepare(root, args.run, settings)
        if args.stage in ("grid", "all"):
            generate(root, directory, data_directory, cfg, options, cache)
        if args.stage in ("analyze", "all"):
            from kerr_sbi.radius_diagnostics import analyze_radius

            analyze_radius(root, directory, data_directory, cfg, options, cache)


if __name__ == "__main__":
    main()
