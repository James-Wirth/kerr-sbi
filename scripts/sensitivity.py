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
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from kerr_sbi.config import DEFAULT_CONFIG, Config, isolate_run, load_config, project_path
from kerr_sbi.fisher import central_difference, fisher_bounds
from kerr_sbi.obs_model import preprocess_pfm
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.simulator import build_scene, render, template_sha256


def render_grid(cfg: Config, config_path: Path) -> dict[str, np.ndarray]:
    settings = cfg["sensitivity"]
    spins = np.linspace(settings["a_min"], settings["a_max"], settings["a_count"])
    inclinations = np.linspace(
        settings["incl_min_deg"], settings["incl_max_deg"], settings["incl_count"]
    )
    delta_a, delta_i = settings["delta_a"], settings["delta_incl_deg"]
    wide_step = delta_a * settings["robustness_step_factor"]
    robustness_indices = np.asarray(settings["robustness_indices"], dtype=int)
    directory = project_path(cfg, "data") / "m2"
    directory.mkdir(parents=True, exist_ok=True)
    cfg["simulator"]["smoke_png"] = False
    tasks = []
    for row, a in enumerate(spins):
        for column, inclination in enumerate(inclinations):
            for role, spin_offset, incl_offset in (
                ("center", 0, 0),
                ("a_minus", -delta_a, 0),
                ("a_plus", delta_a, 0),
                ("i_minus", 0, -delta_i),
                ("i_plus", 0, delta_i),
            ):
                tasks.append((row, column, role, a + spin_offset, inclination + incl_offset))
    for row, column in robustness_indices:
        for role, offset in (("wide_minus", -wide_step), ("wide_plus", wide_step)):
            tasks.append((row, column, role, spins[row] + offset, inclinations[column]))
    for _, _, _, a, inclination in tasks:
        build_scene(float(a), float(inclination), directory / "validation.pfm", cfg)
    metadata = {
        "date_utc": datetime.now(UTC).isoformat(),
        "nullgeo": simulator_provenance(cfg),
        "template_sha256": template_sha256(cfg),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": {key: value for key, value in cfg.items() if key != "project_root"},
        "parameter_order": ["a", "incl_deg"],
        "grid_axis_order": ["a", "incl_deg"],
        "bounds_axis_order": ["noise", "a", "incl_deg", "parameter"],
        "measurement_space": "linear luminance, zero-padded PSF, flux normalization; no asinh",
        "robustness_metric": "L2(wide derivative - nominal derivative) / L2(nominal derivative)",
        "robustness_selection": "four interior grid points; both wider neighbors inside prior",
        "renders": [],
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    images = {}
    for row, column, role, a, inclination in tqdm(tasks, desc="M2 renders", unit="image"):
        stem = directory / f"{row:02d}_{column:02d}_{role}"
        started = time.perf_counter()
        path = render(float(a), float(inclination), stem, cfg)
        elapsed = time.perf_counter() - started
        images[row, column, role] = preprocess_pfm(path, cfg)
        metadata["renders"].append(
            {
                "grid_index": [int(row), int(column)],
                "role": role,
                "a": float(a),
                "incl_deg": float(inclination),
                "pfm_path": str(path.relative_to(cfg["project_root"])),
                "pfm_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "render_seconds": elapsed,
            }
        )
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    grid_shape = (len(spins), len(inclinations))
    image_shape = (cfg["scene"]["height"], cfg["scene"]["width"])
    centers = np.empty((*grid_shape, *image_shape), dtype=np.float32)
    jacobian = np.empty((*grid_shape, np.prod(image_shape), 2), dtype=np.float64)
    for row, column in np.ndindex(grid_shape):
        centers[row, column] = images[row, column, "center"]
        for parameter, role, step in ((0, "a", delta_a), (1, "i", delta_i)):
            jacobian[row, column, :, parameter] = central_difference(
                images[row, column, role + "_minus"],
                images[row, column, role + "_plus"],
                step,
            ).ravel()
    wide_derivatives = np.stack(
        [
            central_difference(
                images[row, column, "wide_minus"], images[row, column, "wide_plus"], wide_step
            ).ravel()
            for row, column in robustness_indices
        ]
    )
    nominal = jacobian[robustness_indices[:, 0], robustness_indices[:, 1], :, 0]
    nominal_norms = np.linalg.norm(nominal, axis=-1)
    relative_changes = np.full_like(nominal_norms, np.inf)
    np.divide(
        np.linalg.norm(wide_derivatives - nominal, axis=-1),
        nominal_norms,
        out=relative_changes,
        where=nominal_norms > 0,
    )
    median_peak = np.median(centers.max(axis=(-2, -1)))
    noise_levels = np.asarray(settings["noise_peak_fractions"]) * median_peak
    bounds = [
        fisher_bounds(jacobian, noise, settings["singular_tolerance"]) for noise in noise_levels
    ]
    return {
        "spins": spins,
        "inclinations_deg": inclinations,
        "centers": centers,
        "jacobian": jacobian,
        "median_peak": np.asarray(median_peak),
        "noise_levels": noise_levels,
        **{key: np.stack([result[key] for result in bounds]) for key in bounds[0]},
        "robustness_indices": robustness_indices,
        "wide_spin_derivatives": wide_derivatives,
        "robustness_relative_changes": relative_changes,
        "robustness_flagged": relative_changes > settings["robustness_relative_tolerance"],
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def save_figure(fig: plt.Figure, directory: Path, name: str, dpi: int) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=dpi)
    plt.close(fig)


def plot_results(data: dict[str, np.ndarray], directory: Path) -> None:
    cfg = json.loads(str(data["metadata_json"]))["config"]
    settings = cfg["sensitivity"]
    dpi = settings["figure_dpi"]
    directory.mkdir(parents=True, exist_ok=True)
    for parameter, label, unit in ((0, "a", "spin"), (1, "i", "degrees")):
        values = np.stack(
            [data[key][..., parameter] for key in ("marginal_std", "conditional_std")]
        )
        finite = values[np.isfinite(values) & (values > 0)]
        if not finite.size:
            raise ValueError(f"no finite {label} bounds to plot")
        norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))
        fig, axes = plt.subplots(
            2, len(data["noise_levels"]), figsize=(14, 6.5), layout="constrained", squeeze=False
        )
        for row, description in enumerate(("Joint fit", "Other parameter known")):
            for column, noise in enumerate(data["noise_levels"]):
                ax = axes[row, column]
                panel = ax.pcolormesh(
                    data["inclinations_deg"],
                    data["spins"],
                    np.ma.masked_invalid(values[row, column]),
                    shading="nearest",
                    norm=norm,
                    cmap="viridis",
                )
                ax.set_facecolor("#bbbbbb")
                ax.set_xlim(data["inclinations_deg"][[0, -1]])
                ax.set_ylim(data["spins"][[0, -1]])
                ax.set_title(
                    f"σn = {noise:.3g} ({settings['noise_peak_fractions'][column]:g} × peak)"
                )
                ax.set_xlabel("Inclination (degrees)")
                if column == 0:
                    ax.set_ylabel(f"{description}\nSpin a")
        fig.colorbar(panel, ax=axes, label=f"σ{label} CRB ({unit}); shared logarithmic scale")
        fig.suptitle(
            f"Local Fisher bounds for {label} · stylized emission · PSF and normalized flux\n"
            "Noise referenced to median grid peak; grey = singular information"
        )
        save_figure(fig, directory, f"m2_crb_{label}", dpi)
    selected = np.asarray(settings["derivative_indices"])
    derivatives = data["jacobian"][selected[:, 0], selected[:, 1], :, 0]
    fig, axes = plt.subplots(2, len(selected), figsize=(11, 7), layout="constrained", squeeze=False)
    for column, (row, col) in enumerate(selected):
        center = data["centers"][row, col]
        scale = cfg["emission_check"]["asinh_peak_fraction"] * center.max()
        axes[0, column].imshow(np.arcsinh(center / scale), cmap="magma", origin="upper")
        axes[0, column].set_title(
            f"a = {data['spins'][row]:.3f}, i = {data['inclinations_deg'][col]:g}°"
        )
        limit = float(np.abs(derivatives[column]).max())
        panel = axes[1, column].imshow(
            derivatives[column].reshape(center.shape),
            cmap="RdBu_r",
            norm=Normalize(-limit, limit),
            origin="upper",
        )
        fig.colorbar(panel, ax=axes[1, column], label="∂x/∂a", shrink=0.8)
        for ax in axes[:, column]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("Preprocessed image\n(per-panel asinh display)")
    axes[1, 0].set_ylabel(f"Spin derivative\nδa = {settings['delta_a']:g}")
    fig.suptitle("Where spin changes the image · independent linear derivative scales")
    save_figure(fig, directory, "m2_spin_derivatives", dpi)


def summarize(data: dict[str, np.ndarray]) -> dict[str, Any]:
    settings = json.loads(str(data["metadata_json"]))["config"]["sensitivity"]

    def number(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    levels = []
    for index, noise in enumerate(data["noise_levels"]):
        marginal = data["marginal_std"][index]
        levels.append(
            {
                "peak_fraction": settings["noise_peak_fractions"][index],
                "sigma_n": float(noise),
                "sigma_a_min_median_max": [
                    number(f(marginal[..., 0])) for f in (np.min, np.median, np.max)
                ],
                "sigma_i_deg_min_median_max": [
                    number(f(marginal[..., 1])) for f in (np.min, np.median, np.max)
                ],
                "well_constrained_grid_points": int(
                    np.sum(marginal[..., 0] < settings["well_constrained_sigma_a"])
                ),
                "weakly_constrained_grid_points": int(
                    np.sum(marginal[..., 0] > settings["weakly_constrained_sigma_a"])
                ),
                "singular_grid_points": int(data["singular"][index].sum()),
            }
        )
    robustness = []
    for index, (row, column) in enumerate(data["robustness_indices"]):
        nominal = data["jacobian"][row, column]
        wide = nominal.copy()
        wide[:, 0] = data["wide_spin_derivatives"][index]
        wide_bounds = fisher_bounds(wide, 1, settings["singular_tolerance"])
        nominal_bounds = fisher_bounds(nominal, 1, settings["singular_tolerance"])
        width_ratio = wide_bounds["marginal_std"] / nominal_bounds["marginal_std"]
        robustness.append(
            {
                "a": float(data["spins"][row]),
                "incl_deg": float(data["inclinations_deg"][column]),
                "relative_derivative_change": number(data["robustness_relative_changes"][index]),
                "wide_to_nominal_sigma_a_ratio": number(width_ratio[0]),
                "wide_to_nominal_sigma_i_ratio": number(width_ratio[1]),
                "flagged": bool(data["robustness_flagged"][index]),
            }
        )
    penalty = data["marginal_std"][0, ..., 0] / data["conditional_std"][0, ..., 0]
    unit_noise_spin_bounds = data["marginal_std"][0, ..., 0] / data["noise_levels"][0]
    return {
        "median_grid_peak": float(data["median_peak"]),
        "grid_points": int(data["centers"].shape[0] * data["centers"].shape[1]),
        "grid_weighting": "uniform a and inclination; counts are not prior probabilities",
        "noise_levels": levels,
        "degeneracy_std_multiplier_min_median_max": [
            number(f(penalty)) for f in (np.min, np.median, np.max)
        ],
        "robustness": robustness,
        "noise_interval_for_requested_spin_span": {
            "sigma_n_lower_exclusive": number(
                settings["weakly_constrained_sigma_a"] / unit_noise_spin_bounds.max()
            ),
            "sigma_n_upper_exclusive": number(
                settings["well_constrained_sigma_a"] / unit_noise_spin_bounds.min()
            ),
            "interpretation": (
                "linear scaling of this local Fisher approximation; not a posterior guarantee"
            ),
        },
        "nonfinite_number_encoding": "null; arrays preserve inf and nan",
        "selection_status": "awaiting user choice at M2 gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.run_name:
        cfg = isolate_run(cfg, args.run_name, plot_only=args.plot_only)
    results = project_path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    archive = results / "crb_grid.npz"
    if args.plot_only:
        with np.load(archive, allow_pickle=False) as saved:
            data = dict(saved)
    else:
        data = render_grid(cfg, args.config)
        np.savez_compressed(archive, **data)
    report = summarize(data)
    (results / "m2_sensitivity.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_results(data, project_path(cfg, "figures"))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
