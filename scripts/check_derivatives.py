import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from tqdm import tqdm

from kerr_sbi.config import DEFAULT_CONFIG, Config, load_config, project_path
from kerr_sbi.fisher import central_difference, fisher_bounds
from kerr_sbi.obs_model import preprocess_pfm
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.simulator import build_scene, render, template_sha256


def validate_baseline(cfg: Config, metadata: dict[str, Any], provenance: dict[str, Any]) -> None:
    if provenance["binary_sha256"] != metadata["nullgeo"]["binary_sha256"]:
        raise ValueError("simulator binary differs from the baseline")
    if template_sha256(cfg) != metadata["template_sha256"]:
        raise ValueError("scene template differs from the baseline")
    previous = metadata["config"]
    for section in ("scene", "emission"):
        if cfg[section] != previous[section]:
            raise ValueError(f"{section} configuration differs from the baseline")
    for key in ("sigma_psf", "kernel_size", "flux_sum"):
        if cfg["observation"][key] != previous["observation"][key]:
            raise ValueError(f"observation {key} differs from the baseline")


def render_checks(cfg: Config, baseline_path: Path) -> dict[str, np.ndarray]:
    with np.load(baseline_path, allow_pickle=False) as archive:
        baseline = dict(archive)
    previous = json.loads(str(baseline["metadata_json"]))
    provenance = simulator_provenance(cfg)
    validate_baseline(cfg, previous, provenance)
    settings = cfg["derivative_check"]
    factors = np.asarray(settings["step_factors"], dtype=np.float64)
    if (
        factors.ndim != 1
        or factors.size < 3
        or not np.isfinite(factors).all()
        or factors[0] != 1
        or np.any(factors <= 0)
        or np.any(np.diff(factors) >= 0)
    ):
        raise ValueError(
            "step_factors must start at one and strictly decrease through positive values"
        )
    sensitivity = previous["config"]["sensitivity"]
    steps = factors[:, None] * [sensitivity["delta_a"], sensitivity["delta_incl_deg"]]
    indices = baseline["robustness_indices"]
    parameters = np.column_stack(
        [baseline["spins"][indices[:, 0]], baseline["inclinations_deg"][indices[:, 1]]]
    )
    nominal = baseline["jacobian"][indices[:, 0], indices[:, 1]]
    jacobians = np.empty((len(factors), *nominal.shape), dtype=np.float64)
    jacobians[0] = nominal
    directory = project_path(cfg, "data") / "m2_convergence"
    directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for level in range(1, len(factors)):
        for point, theta in enumerate(parameters):
            for parameter in range(2):
                for sign in (-1, 1):
                    neighbor = theta.copy()
                    neighbor[parameter] += sign * steps[level, parameter]
                    build_scene(*neighbor, directory / "validation.pfm", cfg)
                    tasks.append((level, point, parameter, sign, neighbor))
    metadata = {
        "date_utc": datetime.now(UTC).isoformat(),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "nullgeo": provenance,
        "template_sha256": template_sha256(cfg),
        "config": {key: value for key, value in cfg.items() if key != "project_root"},
        "parameter_order": ["a", "incl_deg"],
        "jacobian_axis_order": ["step_level", "point", "pixel", "parameter"],
        "relative_change_denominator": "L2 norm of the previous, coarser derivative",
        "diagnostic_noise_status": "comparison only; observation noise is not selected",
        "renders": [],
    }
    images = {}
    for level, point, parameter, sign, theta in tqdm(tasks, desc="Derivative checks", unit="image"):
        stem = directory / f"step{level}_point{point}_parameter{parameter}_{sign:+d}"
        started = time.perf_counter()
        path = render(*theta, stem, cfg)
        elapsed = time.perf_counter() - started
        images[level, point, parameter, sign] = preprocess_pfm(path, cfg)
        metadata["renders"].append(
            {
                "step_level": level,
                "point": point,
                "parameter": parameter,
                "sign": sign,
                "a": float(theta[0]),
                "incl_deg": float(theta[1]),
                "pfm_path": str(path.relative_to(cfg["project_root"])),
                "pfm_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "render_seconds": elapsed,
            }
        )
    for level in range(1, len(factors)):
        for point in range(len(parameters)):
            for parameter in range(2):
                jacobians[level, point, :, parameter] = central_difference(
                    images[level, point, parameter, -1],
                    images[level, point, parameter, 1],
                    steps[level, parameter],
                ).ravel()
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "parameters": parameters,
        "steps": steps,
        "jacobians": jacobians,
        "image_shape": np.asarray(baseline["centers"].shape[-2:]),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def analyze(data: dict[str, np.ndarray]) -> dict[str, Any]:
    cfg = json.loads(str(data["metadata_json"]))["config"]
    settings = cfg["derivative_check"]
    jacobians = data["jacobians"]
    norms = np.linalg.norm(jacobians, axis=-2)
    if np.any(norms == 0):
        raise ValueError("zero derivative norm prevents a relative convergence comparison")
    changes = np.linalg.norm(np.diff(jacobians, axis=0), axis=-2) / norms[:-1]
    cosine = np.sum(jacobians[1:] * jacobians[:-1], axis=-2) / (norms[1:] * norms[:-1])
    bounds = fisher_bounds(
        jacobians, settings["diagnostic_sigma_n"], cfg["sensitivity"]["singular_tolerance"]
    )
    if bounds["singular"].any():
        raise ValueError(
            "singular Fisher information in convergence check; inspect saved Jacobians"
        )
    return {
        "parameters_a_incl_deg": data["parameters"].tolist(),
        "steps_a_incl_deg": data["steps"].tolist(),
        "relative_derivative_changes": changes.tolist(),
        "successive_direction_cosines": np.clip(cosine, -1, 1).tolist(),
        "derivative_norms": norms.tolist(),
        "norm_ratios_to_baseline": (norms / norms[0]).tolist(),
        "marginal_std": bounds["marginal_std"].tolist(),
        "conditional_std": bounds["conditional_std"].tolist(),
        "std_ratios_to_baseline": (bounds["marginal_std"] / bounds["marginal_std"][0]).tolist(),
        "finest_pair_flagged": (changes[-1] > settings["relative_tolerance"]).tolist(),
        "relative_tolerance": settings["relative_tolerance"],
        "diagnostic_sigma_n": settings["diagnostic_sigma_n"],
        "axis_order": "step level (or successive pair), point, parameter (a, incl_deg)",
        "noise_status": "pending user choice at M2 gate",
    }


def save_figures(data: dict[str, np.ndarray], report: dict[str, Any], directory: Path) -> None:
    settings = json.loads(str(data["metadata_json"]))["config"]["derivative_check"]
    directory.mkdir(parents=True, exist_ok=True)
    factors = data["steps"][:, 0] / data["steps"][0, 0]
    changes = np.asarray(report["relative_derivative_changes"])
    ratios = np.asarray(report["std_ratios_to_baseline"])
    fig, axes = plt.subplots(
        2, len(data["parameters"]), figsize=(14, 6), layout="constrained", sharey="row"
    )
    for point, (a, inclination) in enumerate(data["parameters"]):
        axes[0, point].set_title(f"a = {a:.3f}, i = {inclination:g}°")
        for parameter, label in enumerate(("Spin", "Inclination")):
            axes[0, point].plot(factors[1:], changes[:, point, parameter] * 100, "o-", label=label)
            axes[1, point].plot(factors, ratios[:, point, parameter], "o-", label=label)
        axes[0, point].axhline(settings["relative_tolerance"] * 100, color="grey", linestyle="--")
        axes[1, point].axhline(1, color="grey", linestyle="--")
        for ax in axes[:, point]:
            ax.set_xscale("log", base=2)
            ax.invert_xaxis()
            ax.set_xticks(factors, [f"{factor:g}" for factor in factors])
            ax.set_xlabel("Step / original step (smaller →)")
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Change from preceding derivative (%)")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("Joint CRB / original joint CRB")
    axes[0, -1].legend()
    fig.suptitle("Finite-difference stability · fixed renderer, PSF and normalized flux")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"m2_convergence.{suffix}", dpi=settings["figure_dpi"])
    plt.close(fig)
    for parameter, label in enumerate(("a", "incl_deg")):
        fig, axes = plt.subplots(
            len(data["parameters"]), len(factors), figsize=(10, 12), layout="constrained"
        )
        for point, (a, inclination) in enumerate(data["parameters"]):
            derivatives = data["jacobians"][:, point, :, parameter]
            limit = float(np.abs(derivatives).max())
            for level, derivative in enumerate(derivatives):
                ax = axes[point, level]
                panel = ax.imshow(
                    derivative.reshape(data["image_shape"]),
                    cmap="RdBu_r",
                    norm=Normalize(-limit, limit),
                    origin="upper",
                )
                ax.set_title(f"δ = {data['steps'][level, parameter]:g}")
                ax.set_xticks([])
                ax.set_yticks([])
            axes[point, 0].set_ylabel(f"a = {a:.3f}\ni = {inclination:g}°")
            fig.colorbar(panel, ax=axes[point, :], label=f"∂x/∂{label}", shrink=0.8)
        fig.suptitle(f"Derivative with respect to {label} · common linear scale within each row")
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"m2_convergence_{label}.{suffix}", dpi=settings["figure_dpi"])
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = project_path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    archive_path = results / "m2_convergence.npz"
    if args.plot_only:
        with np.load(archive_path, allow_pickle=False) as archive:
            data = dict(archive)
    else:
        data = render_checks(cfg, args.baseline or results / "crb_grid.npz")
        np.savez_compressed(archive_path, **data)
    report = analyze(data)
    (results / "m2_convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    save_figures(data, report, project_path(cfg, "figures"))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
