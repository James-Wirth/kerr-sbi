import argparse
import json
import tomllib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kerr_sbi.config import load_config
from kerr_sbi.data import load_split
from kerr_sbi.persistence import exclusive_lock, file_sha256, write_json
from kerr_sbi.reference import (
    RenderCache,
    compare_mass,
    compare_samples,
    mass_summary,
    relative_log_likelihood,
)
from kerr_sbi.reference_diagnostics import convergence_pass, draw_contours
from kerr_sbi.reference_mesh import ImageMesh, midpoint_points


def physical_points(theta: np.ndarray) -> np.ndarray:
    return np.column_stack([theta[:, 0], np.cos(np.deg2rad(theta[:, 1]))])


def render_points(cache: RenderCache, points: np.ndarray, label: str) -> np.ndarray:
    images = []
    for n, (a, c) in enumerate(points, 1):
        images.append(cache.image(float(a), float(np.rad2deg(np.arccos(c)))))
        if n % 8 == 0 or n == len(points):
            print(f"{label}: {n}/{len(points)}", flush=True)
    return np.stack(images)


def refine(root: Path, source_name: str, run_name: str, settings: dict) -> dict:
    source = root / "runs" / source_name
    destination = root / "runs" / run_name
    data_directory = root / "data" / "reference" / run_name
    experiment = json.loads((source / "experiment.json").read_text())
    previous = json.loads((source / "analysis/summary.json").read_text())
    checkpoint = root / "runs" / experiment["checkpoint"]
    cfg = {**json.loads((checkpoint / "run.json").read_text())["config"], "project_root": root}
    train, dev = load_split(cfg, "train"), load_split(cfg, "dev")
    options = experiment["settings"]["reference"]
    settings = settings["refinement"]
    counts = settings["training_counts"]
    sizes = settings["quadrature_sizes"]
    if (
        len(counts) < 2
        or counts != sorted(set(counts))
        or counts[0] < 1
        or counts[-1] > len(train.x)
        or len(sizes) < 2
        or sizes != sorted(set(sizes))
        or sizes[0] < 2
        or any(b % a for a, b in zip(sizes[:-1], sizes[1:], strict=True))
        or settings["required_successive_passes"] < 1
        or settings["confirmation_per_case"] < 1
        or any(n < 1 for n in settings["new_nodes_per_round"])
    ):
        raise ValueError("invalid refinement counts or nested quadrature sizes")
    grid_size = max(options["grid_sizes"])
    grid_path = root / "data" / "reference" / source_name / f"grid_{grid_size}.npz"
    if file_sha256(grid_path) != previous["grid_sha256"][str(grid_size)]:
        raise ValueError("source grid hash differs from the original report")
    source_archive = source / "analysis/posteriors.npz"
    identity = {
        "source": source_name,
        "settings": settings,
        "thresholds": {key: value for key, value in options.items() if key.startswith("max_")},
        "source_archive_sha256": file_sha256(source_archive),
        "source_summary_sha256": file_sha256(source / "analysis/summary.json"),
        "grid_sha256": file_sha256(grid_path),
        "dataset_arrays": {"train": train.metadata["arrays"], "dev": dev.metadata["arrays"]},
    }
    identity_path = destination / "experiment.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("refinement experiment identity differs")
    write_json(identity_path, identity)
    with np.load(grid_path) as grid:
        base_points = np.stack(np.meshgrid(grid["a"], grid["c"], indexing="ij"), -1).reshape(-1, 2)
        base_images = grid["images"].reshape(-1, 64, 64)
    bounds = np.stack([base_points.min(axis=0), base_points.max(axis=0)])
    with np.load(source_archive) as archive:
        observations, samples = archive["observations"], archive["samples"]
        original_mass = archive["reference_mass"]
    sigma = json.loads((checkpoint / "run.json").read_text())["effective_sigma_n"]
    history, masses = [], []
    consecutive = 0

    def evaluate(mesh: ImageMesh, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal consecutive
        a, c, mass = mesh.mass(observations, sigma, max(sizes))
        entry = {"label": label, "nodes": len(mesh.points)}
        if masses:
            comparison = compare_mass(a, c, masses[-1], mass)
            passed = convergence_pass(comparison, options)
            entry.update(comparison=comparison, passed=passed.tolist())
            consecutive = consecutive + 1 if np.all(passed) else 0
            print(f"{label}: {passed.sum()}/{len(passed)} refinement checks pass", flush=True)
        history.append(entry)
        masses.append(mass)
        write_json(destination / "status.json", {"history": history, "stage": label})
        return a, c, mass

    for count in counts:
        points = np.concatenate([base_points, physical_points(train.theta[:count])])
        images = np.concatenate([base_images, train.x[:count]])
        mesh = ImageMesh(points, images, bounds)
        a, c, mass = evaluate(mesh, f"cached_{count}")
    cache = RenderCache(data_directory / "renders", cfg)
    source_identity = json.loads((grid_path.parent / "renders/identity.json").read_text())
    if cache.identity != source_identity or any(
        split.metadata["nullgeo"] != cache.identity["nullgeo"] for split in (train, dev)
    ):
        raise ValueError("new renders and cached inputs have different forward-model provenance")
    for round_index, count in enumerate(settings["new_nodes_per_round"], 1):
        added = mesh.refinement_points(mass, count)
        points = np.concatenate([points, added])
        images = np.concatenate([images, render_points(cache, added, f"Round {round_index}")])
        mesh = ImageMesh(points, images, bounds)
        a, c, mass = evaluate(mesh, f"round_{round_index}")
        if consecutive >= settings["required_successive_passes"]:
            break
    quadrature_checks = []
    for coarse_size, fine_size in zip(sizes[:-1], sizes[1:], strict=True):
        a0, c0, coarse = mesh.mass(observations, sigma, coarse_size)
        _, _, fine = mesh.mass(observations, sigma, fine_size)
        ratio = fine_size // coarse_size
        rebinned = fine.reshape(-1, coarse_size, ratio, coarse_size, ratio).sum(axis=(2, 4))
        comparison = compare_mass(a0, c0, coarse, rebinned)
        quadrature_checks.append(
            {
                "sizes": [coarse_size, fine_size],
                "comparison": comparison,
                "passed": convergence_pass(comparison, options).tolist(),
            }
        )
    dev_points = physical_points(dev.theta)
    predicted = mesh.interpolate(dev_points)
    dev_norm = np.linalg.norm((predicted - dev.x).reshape(len(dev.x), -1), axis=1) / sigma
    rng = np.random.default_rng(settings["confirmation_seed"])
    _, _, support = midpoint_points(bounds, max(sizes))
    confirmation = np.concatenate(
        [
            support[rng.choice(len(support), settings["confirmation_per_case"], p=row.ravel())]
            for row in mass
        ]
    )
    cell_width = (bounds[1] - bounds[0]) / max(sizes)
    confirmation += rng.uniform(-0.5, 0.5, confirmation.shape) * cell_width
    direct = render_points(cache, confirmation, "Held-out confirmation")
    confirmation_prediction = mesh.interpolate(confirmation)
    confirmation_norm = (
        np.linalg.norm((confirmation_prediction - direct).reshape(len(direct), -1), axis=1) / sigma
    )
    ll_error = relative_log_likelihood(
        observations, confirmation_prediction, sigma
    ) - relative_log_likelihood(observations, direct, sigma)
    confirmation_cases = np.repeat(np.arange(len(mass)), settings["confirmation_per_case"])
    truth_rows = np.array(
        [int(np.flatnonzero(dev.idx == case["dev_idx"])[0]) for case in previous["cases"]]
    )
    truth_ll_error = relative_log_likelihood(
        observations, predicted[truth_rows], sigma
    ) - relative_log_likelihood(observations, dev.x[truth_rows], sigma)
    validation_pass = bool(
        np.all(dev_norm <= options["max_interpolation_noise_norm"])
        and np.all(confirmation_norm <= options["max_interpolation_noise_norm"])
    )
    certified = bool(
        consecutive >= settings["required_successive_passes"]
        and all(all(check["passed"]) for check in quadrature_checks)
        and validation_pass
    )
    report = {
        "identity": identity,
        "cases": previous["cases"],
        "interpolation": "Linear images on a Delaunay mesh in prior-normalized (a, cos i)",
        "history": history,
        "quadrature_checks": quadrature_checks,
        "reference_certified": certified,
        "certification_scope": "Numerical thresholds for selected observations; not calibration",
        "required_successive_passes": settings["required_successive_passes"],
        "consecutive_passes": consecutive,
        "reference_summary": mass_summary(a, c, mass),
        "npe_vs_reference": compare_samples(a, c, mass, samples),
        "interpolation_validation": {
            "development_count": len(dev.x),
            "development_pass_count": int(
                np.sum(dev_norm <= options["max_interpolation_noise_norm"])
            ),
            "development_noise_norm_quantiles_50_90_100": np.quantile(
                dev_norm, [0.5, 0.9, 1]
            ).tolist(),
            "truth_log_likelihood_error": np.diag(truth_ll_error).tolist(),
            "confirmation_count": len(confirmation),
            "confirmation_pass_count": int(
                np.sum(confirmation_norm <= options["max_interpolation_noise_norm"])
            ),
            "confirmation_noise_norm_quantiles_50_90_100": np.quantile(
                confirmation_norm, [0.5, 0.9, 1]
            ).tolist(),
            "confirmation_log_likelihood_error": ll_error[
                confirmation_cases, np.arange(len(confirmation))
            ].tolist(),
            "confirmation_design": (
                "Independent jittered draws from each final interpolated posterior; "
                "never used as mesh nodes"
            ),
        },
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                root / "src/kerr_sbi/reference_mesh.py",
                root / "src/kerr_sbi/reference.py",
                root / "src/kerr_sbi/reference_diagnostics.py",
            )
        },
    }
    np.savez(
        destination / "posteriors.npz",
        a=a,
        c=c,
        reference_mass=mass,
        observations=observations,
        samples=samples,
        masses=np.stack(masses),
        dev_noise_norm=dev_norm,
        confirmation_points=confirmation,
        confirmation_noise_norm=confirmation_norm,
        confirmation_cases=confirmation_cases,
    )
    np.savez(data_directory / "mesh.npz", points=points, images=images, bounds=bounds)
    report["mesh_sha256"] = file_sha256(data_directory / "mesh.npz")
    write_json(destination / "summary.json", report)
    unique_cases = len(options["targets"])
    fig, panels = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for case, panel in enumerate(panels.flat):
        if case >= unique_cases:
            panel.set_axis_off()
            continue
        draw_contours(panel, a, c, mass[case], "#2166ac")
        old_a, old_c, _ = midpoint_points(bounds, original_mass.shape[1])
        draw_contours(panel, old_a, old_c, original_mass[case], "#999999")
        truth = previous["cases"][case]["truth"]
        panel.plot(truth[0], np.cos(np.deg2rad(truth[1])), "k+")
        panel.set(
            xlabel="Spin a",
            ylabel="cos(inclination)",
            title=f"Dev {previous['cases'][case]['dev_idx']}: a={truth[0]:.3f}, i={truth[1]:.1f}°",
        )
    status = "passed numerical checks" if certified else "NOT numerically certified"
    fig.suptitle(f"50% / 90% reference contours · blue: refined · grey: original 17×17\n{status}")
    fig.savefig(destination / "reference_refinement.png", dpi=options["figure_dpi"])
    plt.close(fig)
    write_json(destination / "status.json", {"stage": "complete", "reference_certified": certified})
    print(
        json.dumps(
            {
                "reference_certified": certified,
                "interpolation_validation": report["interpolation_validation"],
            }
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reference_20260926")
    parser.add_argument("--run", required=True)
    parser.add_argument("--settings", type=Path, default=Path("configs/reference_refinement.toml"))
    args = parser.parse_args()
    if args.source == args.run or any(
        not name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in name)
        for name in (args.source, args.run)
    ):
        raise ValueError("source and destination must be distinct valid run names")
    root = load_config()["project_root"]
    destination = root / "runs" / args.run
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(destination / ".lock"):
        refine(root, args.source, args.run, tomllib.loads(args.settings.read_text()))


if __name__ == "__main__":
    main()
