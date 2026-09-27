import argparse
import csv
import json
import os
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kerr_sbi.config import load_config
from kerr_sbi.obs_model import luminance, preprocess
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.pfm import read_pfm
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.radius import RadiusMesh, isco_radius, physical_radius, quadrature_points
from kerr_sbi.radius_diagnostics import refinement_points
from kerr_sbi.radius_triage import failure_table, residual_statistics, simplex_diagnostics
from kerr_sbi.reference import relative_log_likelihood
from kerr_sbi.reference_mesh import ImageMesh
from kerr_sbi.simulator import template_sha256


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def triage(root: Path, source: str, destination: Path) -> None:
    started = time.perf_counter()
    directory, data = root / "runs" / source, root / "data/reference" / source
    report = json.loads((directory / "summary.json").read_text())
    experiment = json.loads((directory / "experiment.json").read_text())
    checkpoint = root / "runs" / experiment["settings"]["radius"]["checkpoint"]
    cfg = {**json.loads((checkpoint / "run.json").read_text())["config"], "project_root": root}
    identity = experiment["render_identity"]
    os.environ.setdefault("NULLGEO_BIN", identity["nullgeo"]["binary"])
    current = {
        "config": {key: cfg[key] for key in identity["config"]},
        "nullgeo": simulator_provenance(cfg),
        "template_sha256": template_sha256(cfg),
        "preprocessing_sha256": file_sha256(root / "src/kerr_sbi/obs_model.py"),
        "pfm_reader_sha256": file_sha256(root / "src/kerr_sbi/pfm.py"),
    }
    if current != identity:
        raise ValueError("current simulation identity differs from cached experiment")
    hashes = {}

    def verify(path: Path, expected: str | None = None) -> None:
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"artifact hash differs: {path}")
        hashes[str(path.relative_to(root))] = actual

    verify(directory / "experiment.json", report["experiment_sha256"])
    verify(data / "mesh.npz", report["mesh_sha256"])
    verify(data / "observations.npz", report["observations_sha256"])
    verify(directory / "posteriors.npz", report["posterior_archive_sha256"])
    for relative, expected in report["source_sha256"].items():
        verify(root / relative, expected)
    for path in [
        directory / "summary.json",
        root / "uv.lock",
        root / "configs/radius.toml",
        Path(__file__),
        root / "src/kerr_sbi/radius_triage.py",
    ]:
        verify(path)
    rows = failure_table(report)
    destination.mkdir(parents=True, exist_ok=False)
    write_table(destination / "failures.csv", rows)
    with np.load(data / "mesh.npz") as archive:
        mesh = RadiusMesh(archive["points"], archive["images"], archive["bounds"])
    with np.load(data / "observations.npz") as archive:
        observations, truths, truth_images = (
            archive["observations"],
            archive["truth"],
            archive["images"],
        )
    with np.load(directory / "posteriors.npz") as archive:
        points, cases, mass = (
            archive["confirmation_points"],
            archive["confirmation_cases"],
            archive["posterior_mass"],
        )
        isco_points = archive["isco_confirmation_points"]
        np.testing.assert_allclose(mass.sum(axis=(2, 3, 4), dtype=float), 1, atol=2e-7)
        np.testing.assert_allclose(mass.sum(axis=(3, 4)), archive["spin_mass"], atol=2e-7)
    radius_max, sigma = report["settings"]["radius_max"], cfg["observation"]["sigma_n"]
    raw_points, raw_paths, raw_hashes = [], [], []
    cache_dirs = [
        data / "renders",
        root / "data/reference/reference_20260926/renders",
        root / "data/reference/reference_20260926/fixed_radius_renders",
        root / "data/reference/reference_refinement_20260926/renders",
    ]
    for cache_dir in cache_dirs:
        cache_identity = json.loads((cache_dir / "identity.json").read_text())
        expected = deepcopy(identity)
        fixed_radius = cache_identity["config"]["scene"]["r_in"]
        expected["config"]["scene"]["r_in"] = fixed_radius
        if cache_identity != expected or fixed_radius not in (0, radius_max):
            raise ValueError(f"incompatible raw cache: {cache_dir}")
        verify(cache_dir / "identity.json")
        for path in sorted(cache_dir.glob("*.json")):
            if path.name == "identity.json":
                continue
            record = json.loads(path.read_text())
            a, inclination, *radius = record["parameters"]
            lower = float(isco_radius(a))
            r = radius[0] if radius else max(fixed_radius, lower)
            raw_points.append(
                [a, np.cos(np.deg2rad(inclination)), (r - lower) / (radius_max - lower)]
            )
            raw_paths.append(path.with_suffix(".pfm"))
            raw_hashes.append(record["sha256"])
            verify(path)
    training = root / "data/datasets/local_pilot_v1/train"
    metadata = json.loads((training / "meta.json").read_text())
    if any(metadata["identity"][key] != value for key, value in identity["config"].items()):
        raise ValueError("training render configuration differs")
    if (
        metadata["nullgeo"] != identity["nullgeo"]
        or metadata["template_sha256"] != identity["template_sha256"]
    ):
        raise ValueError("training renderer identity differs")
    records = [json.loads(line) for line in (training / "renders.jsonl").read_text().splitlines()]
    for record in records:
        if record["status"] == "success":
            raw_points.append([record["a"], np.cos(np.deg2rad(record["incl_deg"])), 0])
            raw_paths.append(training / "pfm" / f"{record['idx']:06d}.pfm")
            raw_hashes.append(record["pfm_sha256"])
    for path in (training / "meta.json", training / "renders.jsonl"):
        verify(path)
    tree = cKDTree(raw_points)
    loaded = {}

    def raw_image(point: np.ndarray) -> np.ndarray:
        distance, index = tree.query(point)
        if distance > 1e-12:
            raise ValueError(f"missing cached raw point: {point}")
        path = raw_paths[index]
        if path not in loaded:
            verify(path, raw_hashes[index])
            loaded[path] = read_pfm(path)
        return loaded[path]

    direct_rgb = np.stack([raw_image(point) for point in points])
    direct = np.stack([preprocess(rgb, cfg) for rgb in direct_rgb])
    prediction = mesh.interpolate(points)
    norms = np.linalg.norm((prediction - direct).reshape(len(points), -1), axis=1) / sigma
    np.testing.assert_allclose(norms, report["validation"]["posterior_noise_norm"], atol=2e-7)
    likelihood_error = relative_log_likelihood(
        observations, prediction, sigma
    ) - relative_log_likelihood(observations, direct, sigma)
    np.testing.assert_allclose(
        likelihood_error[cases, np.arange(len(points))],
        report["validation"]["posterior_log_likelihood_error"],
        atol=2e-7,
    )
    for label, query, direct_images in [
        ("truth", truths, truth_images),
        ("boundary", np.array(report["validation"]["boundary_points"]), None),
    ]:
        checked = np.stack([preprocess(raw_image(point), cfg) for point in query])
        if direct_images is not None:
            np.testing.assert_array_equal(checked, direct_images)
        error = (
            np.linalg.norm((mesh.interpolate(query) - checked).reshape(len(query), -1), axis=1)
            / sigma
        )
        np.testing.assert_allclose(error, report["validation"][f"{label}_noise_norm"], atol=2e-7)
    isco_source = root / "data/reference" / report["settings"]["source_run"] / "mesh.npz"
    verify(isco_source, experiment["source_mesh_sha256"])
    with np.load(isco_source) as archive:
        isco_mesh = ImageMesh(archive["points"], archive["images"], archive["bounds"])
    isco_direct = np.stack(
        [
            preprocess(raw_image(point), cfg)
            for point in np.column_stack([isco_points, np.zeros(len(isco_points))])
        ]
    )
    isco_errors = (
        np.linalg.norm(
            (isco_mesh.interpolate(isco_points) - isco_direct).reshape(len(isco_points), -1), axis=1
        )
        / sigma
    )
    np.testing.assert_allclose(
        isco_errors, report["isco_reference"]["confirmation_noise_norm"], atol=2e-7
    )
    geometry = simplex_diagnostics(mesh, points)
    _, quadrature = quadrature_points(mesh.bounds, list(mass.shape[-3:]))
    quadrature_simplex, _ = mesh.weights(quadrature)
    simplex_mass = np.stack(
        [
            np.bincount(quadrature_simplex, weights=row, minlength=len(mesh.gram))
            for row in mass.reshape(-1, len(quadrature))
        ]
    ).reshape(3, len(observations), -1)
    simplex_score = simplex_mass.max(axis=(0, 1))
    raw_predictions = []
    residual_rows = []
    for index, point in enumerate(points):
        vertices = mesh.triangulation.simplices[geometry["simplex"][index]]
        rgb_vertices = np.stack([raw_image(mesh.points[v]) for v in vertices])
        processed_vertices = np.stack([preprocess(rgb, cfg) for rgb in rgb_vertices])
        np.testing.assert_array_equal(processed_vertices.reshape(4, -1), mesh.images[vertices])
        raw_prediction = np.einsum(
            "v,vij->ij",
            geometry["weights"][index],
            np.stack([luminance(rgb) for rgb in rgb_vertices]),
        )
        raw_predictions.append(raw_prediction)
        raw_residual = raw_prediction - luminance(direct_rgb[index])
        processed_residual = (prediction[index] - direct[index]) / sigma
        row = {
            "check": index,
            "case": int(cases[index]),
            "spin": point[0],
            "inclination_deg": float(np.rad2deg(np.arccos(point[1]))),
            "radius": float(physical_radius(point[None], radius_max)[0]),
        }
        row.update(
            {key: float(values[index]) for key, values in geometry.items() if key != "weights"}
        )
        row.update(
            {f"raw_{key}": value for key, value in residual_statistics(raw_residual).items()}
        )
        row.update(
            {
                f"processed_{key}": value
                for key, value in residual_statistics(processed_residual).items()
            }
        )
        row["log_likelihood_error"] = float(likelihood_error[cases[index], index])
        score = simplex_score[geometry["simplex"][index]]
        row["selector_mass_score"] = float(score)
        row["selector_mass_rank"] = int(1 + np.sum(simplex_score > score))
        for prior, name in enumerate(report["prior_names"]):
            row[f"{name}_simplex_mass"] = float(
                simplex_mass[prior, cases[index], geometry["simplex"][index]]
            )
        residual_rows.append(row)
    write_table(destination / "posterior_residuals.csv", residual_rows)
    audit = {}
    for count in (128, 192):
        candidates = refinement_points(mesh, mass, quadrature, count)
        distance, _ = cKDTree((mesh.points - mesh.bounds[0]) / mesh.width).query(
            (candidates - mesh.bounds[0]) / mesh.width
        )
        audit[str(count)] = {
            "count": len(candidates),
            "unique": len(np.unique(candidates, axis=0)),
            "minimum_existing_node_distance": float(distance.min()),
            "isco_face_candidates": int(np.sum(candidates[:, 2] == 0)),
            "upper_radius_face_candidates": int(np.sum(candidates[:, 2] == 1)),
            "candidates_radius_fraction_below_0_05": int(np.sum(candidates[:, 2] < 0.05)),
        }
        np.save(destination / f"mass_only_candidates_{count}.npy", candidates)
    np.savez(
        destination / "residuals.npz",
        points=points,
        cases=cases,
        raw_direct=np.stack([luminance(rgb) for rgb in direct_rgb]),
        raw_prediction=np.stack(raw_predictions),
        processed_direct=direct,
        processed_prediction=prediction,
        isco_points=isco_points,
        isco_direct=isco_direct,
        isco_prediction=isco_mesh.interpolate(isco_points),
    )
    selected_cases = [18, 10, 6, 14, 22, 19, 0, 21]
    fig, axes = plt.subplots(
        len(selected_cases), 4, figsize=(12, 2.5 * len(selected_cases)), layout="constrained"
    )
    for row_index, case in enumerate(selected_cases):
        indices = np.flatnonzero(cases == case)
        index = indices[np.argmax(norms[indices])]
        raw = luminance(direct_rgb[index])
        arrays = [
            raw,
            raw_predictions[index] - raw,
            direct[index],
            (prediction[index] - direct[index]) / sigma,
        ]
        titles = [
            "Raw linear luminance",
            "Raw interpolation residual",
            "PSF + flux normalization",
            "Residual / pixel noise",
        ]
        for col, (array, title) in enumerate(zip(arrays, titles, strict=True)):
            limit = float(np.abs(array).max())
            artist = axes[row_index, col].imshow(
                array,
                cmap="RdBu_r" if col % 2 else "magma",
                vmin=-limit if col % 2 else 0,
                vmax=limit,
            )
            axes[row_index, col].set_title(f"Case {case}, check {index}\n{title}", fontsize=9)
            axes[row_index, col].set_xticks([])
            axes[row_index, col].set_yticks([])
            fig.colorbar(artist, ax=axes[row_index, col], shrink=0.7)
    fig.savefig(destination / "residuals.png", dpi=130)
    plt.close(fig)
    audit["singular_tetrahedra"] = int(
        np.sum(~np.isfinite(mesh.triangulation.transform).all(axis=(1, 2)))
    )
    audit["confirmation_mesh_minimum_distance"] = float(cKDTree(mesh.points).query(points)[0].min())
    audit["boundary_rule"] = (
        "Existing selector excludes the t=0 ISCO face; "
        "other hull faces receive one quarter of candidates."
    )
    summary = {
        "accepted": sum(row["accepted"] for row in rows),
        "isco_accepted": int(np.sum(report["isco_reference"]["certified"])),
        "failed_checks": {
            name: sum(not row[f"{name}_pass"] for row in rows)
            for name in ("forward", "quadrature", "truth", "posterior", "boundary")
        },
        "unresolved_isco_cases": np.flatnonzero(
            ~np.array(report["isco_reference"]["certified"])
        ).tolist(),
        "high_spin_high_inclination_isco_cases": [6, 14, 22],
        "regression_anchor": 19,
        "max_posterior_noise_norm": float(norms.max()),
        "mesh_audit": audit,
        "inspected_images": {
            "truths": len(truth_images),
            "posterior_checks": len(direct),
            "isco_checks": len(isco_direct),
            "max_processed_pixel": float(max(truth_images.max(), direct.max(), isco_direct.max())),
            "sigma_n": sigma,
            "images_with_pixel_at_least_sigma": int(
                sum(
                    np.sum(images.max(axis=(1, 2)) >= sigma)
                    for images in (truth_images, direct, isco_direct)
                )
            ),
        },
        "new_renders": 0,
        "analysis_seconds": time.perf_counter() - started,
    }
    write_json(destination / "summary.json", summary)
    write_json(
        destination / "manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "working_tree_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=root, text=True
            ),
            "simulation_identity": identity,
            "library_versions": {
                name: version(name) for name in ("numpy", "scipy", "jax", "matplotlib")
            },
            "input_sha256": hashes,
            "budget": {
                "render_cap": 640,
                "renderer_seconds_cap": 7200,
                "integration_seconds_cap": 7200,
                "used_renders": 0,
                "used_renderer_seconds": 0,
                "allocation": {
                    "diagnostics": 128,
                    "refinement": 320,
                    "fresh_posterior": 96,
                    "boundary_isco_followup": 96,
                },
            },
            "confirmation_status": (
                "Original confirmation points are now development diagnostics; "
                "fresh acceptance points required after adaptation."
            ),
            "scope": (
                "Cached triage. Acceptance flags reconstructed from saved numerical comparisons "
                "and independently reproduced direct-render errors; no quadrature recomputation."
            ),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify numerical failures of a cached radius reference without rendering"
    )
    parser.add_argument(
        "--source", required=True, help="radius reference run, e.g. radius_20260926"
    )
    parser.add_argument("--run", required=True, help="fresh output: runs/RUN")
    args = parser.parse_args()
    root = load_config()["project_root"]
    for name in (args.source, args.run):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("source and run must be directory names")
    triage(root, args.source, root / "runs" / args.run)
