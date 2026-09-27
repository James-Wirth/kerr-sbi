import argparse
import hashlib
import json
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from tqdm import tqdm

from kerr_sbi.config import DEFAULT_CONFIG, Config, isolate_run, load_config, project_path
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


def difference_steps(cfg: Config, baseline_config: Config) -> np.ndarray:
    factors = np.asarray(cfg["derivative_check"]["step_factors"], dtype=np.float64)
    if (
        factors.ndim != 1
        or factors.size < 2
        or not np.isfinite(factors).all()
        or factors[0] != 1
        or np.any(factors <= 0)
        or np.any(np.diff(factors) >= 0)
    ):
        raise ValueError(
            "step_factors must start at one and strictly decrease through positive values"
        )
    sensitivity = baseline_config["sensitivity"]
    return factors[:, None] * [sensitivity["delta_a"], sensitivity["delta_incl_deg"]]


def render_checks(
    cfg: Config,
    baseline_path: Path,
    supersample: int | None = None,
    grid_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    with np.load(baseline_path, allow_pickle=False) as archive:
        baseline = dict(archive)
    previous = json.loads(str(baseline["metadata_json"]))
    provenance = simulator_provenance(cfg)
    validate_baseline(cfg, previous, provenance)
    cfg = deepcopy(cfg)
    active_supersample = cfg["scene"]["supersample"]
    if supersample is not None:
        if not isinstance(supersample, int) or supersample < 1:
            raise ValueError("supersample must be a positive integer")
        cfg["scene"]["supersample"] = supersample
    steps = difference_steps(cfg, previous["config"])
    indices = baseline["robustness_indices"] if grid_indices is None else grid_indices
    parameters = np.column_stack(
        [baseline["spins"][indices[:, 0]], baseline["inclinations_deg"][indices[:, 1]]]
    )
    nominal = baseline["jacobian"][indices[:, 0], indices[:, 1]]
    jacobians = np.empty((len(steps), *nominal.shape), dtype=np.float64)
    reuse_nominal = cfg["scene"]["supersample"] == previous["config"]["scene"]["supersample"]
    if reuse_nominal:
        jacobians[0] = nominal
    directory = project_path(cfg, "data") / (
        "m2_convergence" if supersample is None else f"m2_supersampling/ss{supersample}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    first_level = 1 if reuse_nominal else 0
    for level in range(first_level, len(steps)):
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
        "baseline_supersample": previous["config"]["scene"]["supersample"],
        "active_supersample": active_supersample,
        "sampling_override": supersample,
        "nominal_derivative_reused": reuse_nominal,
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
    for level in range(first_level, len(steps)):
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


def save_figures(
    data: dict[str, np.ndarray],
    report: dict[str, Any],
    directory: Path,
    prefix: str = "m2_convergence",
) -> None:
    cfg = json.loads(str(data["metadata_json"]))["config"]
    settings = cfg["derivative_check"]
    sampling = cfg["scene"]["supersample"]
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
    fig.suptitle(
        f"Finite-difference stability · {sampling}×{sampling} subpixel rays · fixed PSF and flux"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{prefix}.{suffix}", dpi=settings["figure_dpi"])
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
        fig.suptitle(
            f"Derivative with respect to {label} · {sampling}×{sampling} subpixel rays\n"
            "Common linear scale within each row"
        )
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{prefix}_{label}.{suffix}", dpi=settings["figure_dpi"])
        plt.close(fig)


def reuse_checks(
    cfg: Config, baseline_path: Path, reference_path: Path, supersample: int
) -> dict[str, np.ndarray]:
    with np.load(baseline_path, allow_pickle=False) as archive:
        baseline = dict(archive)
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = dict(archive)
    previous = json.loads(str(baseline["metadata_json"]))
    source = json.loads(str(reference["metadata_json"]))
    provenance = simulator_provenance(cfg)
    validate_baseline(cfg, previous, provenance)
    sampling_config = deepcopy(cfg)
    sampling_config["scene"]["supersample"] = supersample
    validate_baseline(sampling_config, source, provenance)
    if source["baseline_sha256"] != hashlib.sha256(baseline_path.read_bytes()).hexdigest():
        raise ValueError("reference belongs to a different baseline archive")
    if source["parameter_order"] != ["a", "incl_deg"] or source["jacobian_axis_order"] != [
        "step_level",
        "point",
        "pixel",
        "parameter",
    ]:
        raise ValueError("reference parameter or Jacobian axis order differs")
    indices = baseline["robustness_indices"]
    parameters = np.column_stack(
        [baseline["spins"][indices[:, 0]], baseline["inclinations_deg"][indices[:, 1]]]
    )
    image_shape = np.asarray(baseline["centers"].shape[-2:])
    if not np.array_equal(reference["parameters"], parameters):
        raise ValueError("reference parameters differ from the baseline diagnostic points")
    if not np.array_equal(reference["image_shape"], image_shape):
        raise ValueError("reference image shape differs from the baseline")
    recorded_steps = difference_steps(source["config"], previous["config"])
    if not np.array_equal(reference["steps"], recorded_steps):
        raise ValueError("reference steps differ from its recorded configuration")
    expected_shape = (len(recorded_steps), len(parameters), int(image_shape.prod()), 2)
    if (
        reference["jacobians"].shape != expected_shape
        or not np.isfinite(reference["jacobians"]).all()
    ):
        raise ValueError("reference Jacobians have invalid shape or values")
    steps = difference_steps(cfg, previous["config"])
    selected = []
    for step in steps:
        matches = np.flatnonzero(np.all(reference["steps"] == step, axis=1))
        if len(matches) != 1:
            raise ValueError(f"reference does not contain the requested step: {step}")
        selected.append(int(matches[0]))
    cases = {}
    for case in source["renders"]:
        key = (case["step_level"], case["point"], case["parameter"], case["sign"])
        if key in cases:
            raise ValueError("reference contains duplicate render stencils")
        cases[key] = case
    selected_renders = []
    for level, source_level in enumerate(selected):
        for point, theta in enumerate(parameters):
            for parameter in range(2):
                if source_level == 0 and supersample == previous["config"]["scene"]["supersample"]:
                    nominal = baseline["jacobian"][*indices[point], :, parameter]
                    if not np.array_equal(
                        reference["jacobians"][source_level, point, :, parameter], nominal
                    ):
                        raise ValueError("reused nominal derivative differs from the baseline")
                    continue
                images = []
                for sign in (-1, 1):
                    key = (source_level, point, parameter, sign)
                    if key not in cases:
                        raise ValueError("reference is missing a requested render stencil")
                    case = cases[key]
                    neighbor = theta.copy()
                    neighbor[parameter] += sign * steps[level, parameter]
                    if not np.array_equal([case["a"], case["incl_deg"]], neighbor):
                        raise ValueError(
                            "reference render parameters differ from the requested stencil"
                        )
                    path = Path(cfg["project_root"]) / case["pfm_path"]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != case["pfm_sha256"]:
                        raise ValueError(f"reference PFM hash differs: {path}")
                    images.append(preprocess_pfm(path, sampling_config))
                    selected_renders.append({**case, "step_level": level})
                computed = central_difference(*images, steps[level, parameter]).ravel()
                if not np.array_equal(
                    computed, reference["jacobians"][source_level, point, :, parameter]
                ):
                    raise ValueError("reference derivative differs from its preprocessed PFMs")
    metadata = {
        **source,
        "config": {key: value for key, value in sampling_config.items() if key != "project_root"},
        "source_archive": str(reference_path.resolve()),
        "source_archive_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_metadata": source,
        "source_step_indices": selected,
        "renders": selected_renders,
        "archive_reused": True,
    }
    return {
        "parameters": parameters,
        "steps": steps,
        "image_shape": image_shape,
        "jacobians": reference["jacobians"][selected],
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def render_supersampling(
    cfg: Config, baseline_path: Path, reference_paths: list[Path] | None = None
) -> dict[str, np.ndarray]:
    levels = cfg["supersampling_check"]["levels"]
    if (
        len(levels) < 2
        or any(not isinstance(level, int) or level < 1 for level in levels)
        or any(high <= low for low, high in zip(levels[:-1], levels[1:], strict=True))
    ):
        raise ValueError("sampling levels must be positive integers and strictly increase")
    results = project_path(cfg, "results")
    references = {}
    for path in (
        reference_paths if reference_paths is not None else [results / "m2_convergence.npz"]
    ):
        with np.load(path, allow_pickle=False) as archive:
            level = json.loads(str(archive["metadata_json"]))["config"]["scene"]["supersample"]
        if level not in levels or level in references:
            raise ValueError("reference sampling must be requested and unique")
        references[level] = reuse_checks(cfg, baseline_path, path, level)
    runs = []
    for level in levels:
        data = references.get(level)
        if data is None:
            data = render_checks(cfg, baseline_path, supersample=level)
        if runs:
            for key in ("parameters", "steps", "image_shape"):
                if not np.array_equal(data[key], runs[0][key]):
                    raise ValueError(f"{key} differ between sampling runs")
        results.mkdir(parents=True, exist_ok=True)
        destination = results / f"m2_supersampling_{level}.npz"
        if destination.resolve() in {path.resolve() for path in reference_paths or []}:
            raise ValueError("output archive would overwrite a reused reference")
        np.savez_compressed(destination, **data)
        runs.append(data)
    return {
        "supersamples": np.asarray(levels),
        "parameters": runs[0]["parameters"],
        "steps": runs[0]["steps"],
        "image_shape": runs[0]["image_shape"],
        "jacobians": np.stack([run["jacobians"] for run in runs]),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "config": {key: value for key, value in cfg.items() if key != "project_root"},
                    "runs": [json.loads(str(run["metadata_json"])) for run in runs],
                }
            )
        ),
    }


def sampling_run(data: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {
        "parameters": data["parameters"],
        "steps": data["steps"],
        "image_shape": data["image_shape"],
        "jacobians": data["jacobians"][index],
        "metadata_json": np.asarray(
            json.dumps(json.loads(str(data["metadata_json"]))["runs"][index])
        ),
    }


def analyze_supersampling(data: dict[str, np.ndarray]) -> dict[str, Any]:
    metadata = json.loads(str(data["metadata_json"]))
    reports = [analyze(sampling_run(data, index)) for index in range(len(data["supersamples"]))]
    norms = np.linalg.norm(data["jacobians"], axis=-2)
    sampling_changes = np.linalg.norm(np.diff(data["jacobians"], axis=0), axis=-2) / norms[:-1]
    bounds = np.asarray([report["marginal_std"] for report in reports])
    render_times = []
    for run in metadata["runs"]:
        timings = {}
        for case in run["renders"]:
            key = (
                *data["steps"][case["step_level"]],
                *data["parameters"][case["point"]],
                case["parameter"],
                case["sign"],
            )
            if (
                key in timings
                or not np.isfinite(case["render_seconds"])
                or case["render_seconds"] <= 0
            ):
                raise ValueError("render timing stencils must be unique with finite positive times")
            timings[key] = case["render_seconds"]
        render_times.append(timings)
    matched = set.intersection(*(set(timings) for timings in render_times))
    if not matched:
        raise ValueError("sampling runs have no matching timed render stencils")
    timings = [[run[key] for key in sorted(matched)] for run in render_times]
    medians = np.array([np.median(times) for times in timings])
    cfg = metadata["runs"][0]["config"]
    derivative_tolerance = cfg["derivative_check"]["relative_tolerance"]
    bound_tolerance = cfg["supersampling_check"].get("joint_relative_tolerance")
    step_changes = np.asarray([report["relative_derivative_changes"] for report in reports])
    step_bound_changes = np.abs(bounds[:, 1:] / bounds[:, :-1] - 1)
    sampling_bound_changes = np.abs(bounds[1:] / bounds[:-1] - 1)
    flags = {
        "step_derivative_flagged": step_changes >= derivative_tolerance,
        "sampling_derivative_flagged": sampling_changes >= derivative_tolerance,
    }
    if bound_tolerance is not None:
        if not np.isfinite(bound_tolerance) or bound_tolerance <= 0:
            raise ValueError("joint bound tolerance must be finite and positive")
        flags["step_joint_bound_flagged"] = step_bound_changes >= bound_tolerance
        flags["sampling_joint_bound_flagged"] = sampling_bound_changes >= bound_tolerance
    return {
        "supersamples": data["supersamples"].tolist(),
        "rays_per_pixel": (data["supersamples"] ** 2).tolist(),
        "levels": reports,
        "cross_sampling_derivative_changes": sampling_changes.tolist(),
        "cross_sampling_direction_cosines": np.clip(
            np.sum(data["jacobians"][1:] * data["jacobians"][:-1], axis=-2)
            / (norms[1:] * norms[:-1]),
            -1,
            1,
        ).tolist(),
        "cross_sampling_std_ratios": (bounds[1:] / bounds[:-1]).tolist(),
        "std_ratios_to_original_sampling": (bounds / bounds[0]).tolist(),
        "cross_sampling_axis_order": "successive sampling pair, step level, point, parameter",
        "matched_render_counts": [len(times) for times in timings],
        "matched_median_render_seconds": medians.tolist(),
        "matched_median_render_time_ratios": (medians / medians[0]).tolist(),
        "render_timing_scope": (
            "intersection of identical parameter, step, parameter-column and sign stencils; "
            "reused archives retain historical timings"
        ),
        "acceptance": {
            "derivative_relative_tolerance": derivative_tolerance,
            "joint_bound_relative_tolerance": bound_tolerance,
            "strictly_below_tolerances": True,
            **{key: value.tolist() for key, value in flags.items()},
            "all_passed": not any(value.any() for value in flags.values()),
            "scope": "tested points and steps only; not prior-wide or infinitesimal convergence",
        },
        "sampling_status": "diagnostic comparison; active v1 sampling is unchanged",
        "noise_status": "pending user choice at M2 gate",
    }


def save_sampling_figures(
    data: dict[str, np.ndarray], report: dict[str, Any], directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(str(data["metadata_json"]))["config"]
    levels = data["supersamples"]
    changes = np.asarray([level["relative_derivative_changes"][-1] for level in report["levels"]])
    ratios = np.asarray(report["std_ratios_to_original_sampling"])[:, -1]
    sampling_changes = np.asarray(report["cross_sampling_derivative_changes"])[:, -1]
    fig, axes = plt.subplots(
        3, 2, figsize=(11, 10), layout="constrained", sharex=True, sharey="row"
    )
    for parameter, label in enumerate(("Spin", "Inclination")):
        axes[0, parameter].set_title(label)
        for point, (a, inclination) in enumerate(data["parameters"]):
            point_label = f"a={a:.3f}, i={inclination:g}°"
            axes[0, parameter].plot(
                levels, 100 * changes[:, point, parameter], "o-", label=point_label
            )
            axes[1, parameter].plot(levels[1:], 100 * sampling_changes[:, point, parameter], "o-")
            axes[2, parameter].plot(levels, ratios[:, point, parameter], "o-")
        axes[0, parameter].axhline(
            cfg["derivative_check"]["relative_tolerance"] * 100, color="grey", linestyle="--"
        )
        axes[1, parameter].axhline(
            cfg["derivative_check"]["relative_tolerance"] * 100, color="grey", linestyle="--"
        )
        axes[2, parameter].axhline(1, color="grey", linestyle="--")
        for ax in axes[:, parameter]:
            ax.set_xticks(levels, [f"{n}×{n}" for n in levels])
            ax.grid(alpha=0.2)
        axes[2, parameter].set_xlabel("Subpixel rays per pixel")
    axes[0, 0].set_ylabel("Finest step-pair derivative change (%)")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("Finest derivative change from\npreceding sampling level (%)")
    axes[1, 0].set_ylim(bottom=0)
    axes[2, 0].set_ylabel(f"Finest-step joint CRB / {levels[0]}×{levels[0]} joint CRB")
    axes[0, 1].legend(fontsize=9)
    fig.suptitle("Supersampling check · 64×64 images · fixed physics and preprocessing")
    for suffix in ("png", "pdf"):
        fig.savefig(
            directory / f"m2_supersampling.{suffix}", dpi=cfg["derivative_check"]["figure_dpi"]
        )
    plt.close(fig)
    for index, level in enumerate(levels[1:], start=1):
        save_figures(
            sampling_run(data, index),
            report["levels"][index],
            directory,
            f"m2_supersampling_{level}",
        )
    for parameter, label in enumerate(("a", "incl_deg")):
        for lower in range(len(levels) - 1):
            fig, axes = plt.subplots(
                len(data["parameters"]),
                len(data["steps"]),
                figsize=(10, 10),
                layout="constrained",
                squeeze=False,
            )
            differences = (
                data["jacobians"][lower + 1, ..., parameter]
                - data["jacobians"][lower, ..., parameter]
            )
            for point, (a, inclination) in enumerate(data["parameters"]):
                limit = max(float(np.abs(differences[:, point]).max()), np.finfo(float).tiny)
                for step, values in enumerate(differences[:, point]):
                    ax = axes[point, step]
                    panel = ax.imshow(
                        values.reshape(data["image_shape"]),
                        cmap="RdBu_r",
                        norm=Normalize(-limit, limit),
                        origin="upper",
                    )
                    ax.set_title(f"δ = {data['steps'][step, parameter]:g}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                axes[point, 0].set_ylabel(f"a = {a:.3f}\ni = {inclination:g}°")
                fig.colorbar(
                    panel, ax=axes[point, :], label=f"Difference in ∂x/∂{label}", shrink=0.8
                )
            fig.suptitle(
                f"Derivative differences · {levels[lower + 1]}×{levels[lower + 1]} minus "
                f"{levels[lower]}×{levels[lower]}\nCommon linear scale within each row"
            )
            for suffix in ("png", "pdf"):
                fig.savefig(
                    directory
                    / f"m2_difference_{levels[lower]}_{levels[lower + 1]}_{label}.{suffix}",
                    dpi=cfg["derivative_check"]["figure_dpi"],
                )
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check derivative convergence in step size and supersampling"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--supersampling", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--step-factors", type=float, nargs="+")
    parser.add_argument("--reuse", type=Path, action="append")
    parser.add_argument("--run-name")
    parser.add_argument("--grid-corners", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    baseline_path = args.baseline or project_path(cfg, "results") / "crb_grid.npz"
    reference_paths = args.reuse or [project_path(cfg, "results") / "m2_convergence.npz"]
    if (args.levels or args.reuse) and not args.supersampling:
        parser.error("--levels and --reuse require --supersampling")
    if args.grid_corners and args.supersampling:
        parser.error("--grid-corners is a within-sampling check")
    if (args.levels or args.reuse or args.step_factors or args.grid_corners) and not args.run_name:
        parser.error("custom comparisons require --run-name to preserve previous outputs")
    if args.levels:
        cfg["supersampling_check"]["levels"] = args.levels
    if args.step_factors:
        cfg["derivative_check"]["step_factors"] = args.step_factors
    if args.run_name:
        cfg = isolate_run(cfg, args.run_name, plot_only=args.plot_only)
    results = project_path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    name = "m2_supersampling" if args.supersampling else "m2_convergence"
    archive_path = results / f"{name}.npz"
    if args.plot_only:
        with np.load(archive_path, allow_pickle=False) as archive:
            data = dict(archive)
    else:
        if args.supersampling:
            data = render_supersampling(cfg, baseline_path, reference_paths)
        else:
            grid_indices = None
            if args.grid_corners:
                with np.load(baseline_path, allow_pickle=False) as baseline:
                    last_a, last_i = (
                        len(baseline["spins"]) - 1,
                        len(baseline["inclinations_deg"]) - 1,
                    )
                grid_indices = np.array([[0, 0], [0, last_i], [last_a, 0], [last_a, last_i]])
            data = render_checks(cfg, baseline_path, grid_indices=grid_indices)
        np.savez_compressed(archive_path, **data)
    report = analyze_supersampling(data) if args.supersampling else analyze(data)
    (results / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot = save_sampling_figures if args.supersampling else save_figures
    plot(data, report, project_path(cfg, "figures"))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
