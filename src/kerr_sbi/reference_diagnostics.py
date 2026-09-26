import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kerr_sbi.data import load_split, prior_log_prob, theta_to_u, u_to_theta
from kerr_sbi.inference import load_posterior
from kerr_sbi.obs_model import add_noise, network_transform
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.reference import (
    compare_mass,
    compare_samples,
    interpolate_images,
    mass_summary,
    prior_particle_reference,
    reference_mass,
    relative_log_likelihood,
)


def select_cases(truth: np.ndarray, targets: list, bounds: tuple) -> np.ndarray:
    if not 1 <= len(targets) <= len(truth):
        raise ValueError("targets must fit inside the development set")
    physical = np.column_stack([truth[:, 0], np.cos(np.deg2rad(truth[:, 1]))])
    selected = []
    width = np.array([axis[-1] - axis[0] for axis in bounds])
    for a, inclination in targets:
        point = np.array([a, np.cos(np.deg2rad(inclination))])
        distance = np.sum(((physical - point) / width) ** 2, axis=1)
        distance[selected] = np.inf
        selected.append(int(np.argmin(distance)))
    return np.array(selected)


def convergence_pass(comparison: dict, settings: dict) -> np.ndarray:
    return np.all(
        np.stack(
            [
                np.asarray(comparison[name]) <= settings[f"max_{name}"]
                for name in ("marginal_cdf_change", "mean_shift_in_std", "relative_std_change")
            ]
        ),
        axis=(0, 2),
    )


def draw_contours(
    axis: plt.Axes, a: np.ndarray, c: np.ndarray, mass: np.ndarray, color: str
) -> None:
    descending = np.sort(mass.ravel())[::-1]
    cumulative = np.cumsum(descending)
    levels = sorted(
        {
            float(descending[min(np.searchsorted(cumulative, p), len(descending) - 1)])
            for p in (0.5, 0.9)
        }
    )
    axis.contour(a, c, mass.T, levels=levels, colors=color, linewidths=1.2)


def analyze(
    directory: Path, checkpoint: Path, settings: dict, *, grid_directory: Path | None = None
) -> dict:
    grid_directory = directory if grid_directory is None else grid_directory
    options = settings["reference"]
    posterior = load_posterior(checkpoint)
    cfg = {**posterior.identity["config"], "project_root": checkpoint.parent.parent}
    data = load_split(cfg, "dev")
    grids = {}
    for size in options["grid_sizes"]:
        with np.load(grid_directory / f"grid_{size}.npz") as archive:
            grids[size] = ((archive["a"], archive["c"]), archive["images"])
    axes, fine_images = grids[max(grids)]
    selected = select_cases(data.theta, options["targets"], axes)
    row_indices, noise_seeds, observations = [], [], []
    sigma = posterior.identity["effective_sigma_n"]
    for seed in options["noise_seeds"]:
        for row in selected:
            key = jax.random.fold_in(jax.random.PRNGKey(seed), int(data.idx[row]))
            observations.append(np.asarray(add_noise(jnp.asarray(data.x[row]), key, sigma)))
            row_indices.append(int(row))
            noise_seeds.append(seed)
    observations = np.stack(observations)
    quadrature = max(options["quadrature_sizes"])
    masses = {}
    for size, (grid_axes, grid_images) in grids.items():
        print(
            f"Integrating {size}x{size} interpolator on {quadrature}x{quadrature} cells", flush=True
        )
        a, c, masses[size] = reference_mass(grid_axes, grid_images, observations, sigma, quadrature)
    reference = masses[max(grids)]
    grid_comparison = compare_mass(a, c, masses[min(grids)], reference)
    coarse_size = min(options["quadrature_sizes"])
    if quadrature % coarse_size:
        raise ValueError("quadrature sizes must divide")
    a0, c0, coarse = reference_mass(axes, fine_images, observations, sigma, coarse_size)
    ratio = quadrature // coarse_size
    rebinned = reference.reshape(-1, coarse_size, ratio, coarse_size, ratio).sum(axis=(2, 4))
    quadrature_comparison = compare_mass(a0, c0, coarse, rebinned)
    z = network_transform(jnp.asarray(observations), posterior.normalization.asinh_scale)[:, None]
    sample = eqx.filter_jit(
        lambda model, z, key: model.sample(z, key, options["posterior_samples"])
    )
    draws = sample(posterior.model, z, jax.random.key(options["posterior_seed"]))
    samples = np.asarray(u_to_theta(draws, posterior.normalization))
    points = np.stack(np.meshgrid(a, c, indexing="ij"), axis=-1).reshape(-1, 2)
    theta_points = np.column_stack([points[:, 0], np.rad2deg(np.arccos(points[:, 1]))])
    u = theta_to_u(jnp.asarray(theta_points), posterior.normalization)
    log_prior = np.asarray(prior_log_prob(u, posterior.normalization))
    embedding = jax.vmap(posterior.model.embedding)(z)
    log_prob = eqx.filter_jit(
        lambda flow, values, condition: flow.log_prob(values, condition=condition)
    )
    npe_mass = np.empty_like(reference)
    npe_integral = []
    for case in range(len(z)):
        log_ratios = []
        for start in range(0, len(points), 4096):
            log_ratios.append(
                np.asarray(log_prob(posterior.model.flow, u[start : start + 4096], embedding[case]))
                - log_prior[start : start + 4096]
            )
        density_ratio = np.exp(np.concatenate(log_ratios).astype(np.float64))
        npe_integral.append(float(density_ratio.mean()))
        npe_mass[case] = (density_ratio / density_ratio.sum()).reshape(quadrature, quadrature)
    rng = np.random.default_rng(options["validation_seed"])
    validation = np.unique(
        np.r_[
            selected,
            rng.choice(
                len(data.idx), min(options["validation_count"], len(data.idx)), replace=False
            ),
        ]
    )
    validation_points = np.column_stack(
        [data.theta[validation, 0], np.cos(np.deg2rad(data.theta[validation, 1]))]
    )
    interpolated = interpolate_images(axes, fine_images, validation_points)
    residual = (interpolated - data.x[validation]) / sigma
    noise_norm = np.linalg.norm(residual.reshape(len(validation), -1), axis=1)
    interpolation_errors = relative_log_likelihood(
        observations, interpolated, sigma
    ) - relative_log_likelihood(observations, data.x[validation], sigma)
    truth_columns = np.array([int(np.flatnonzero(validation == row)[0]) for row in row_indices])
    truth_errors = interpolation_errors[np.arange(len(observations)), truth_columns]
    reference_summary = mass_summary(a, c, reference)
    comparison = compare_samples(a, c, reference, samples)
    training = load_split(cfg, "train")
    particle_reference = prior_particle_reference(observations, training.x, training.theta, sigma)
    report = {
        "checkpoint": json.loads((checkpoint / "latest.json").read_text()),
        "settings": settings,
        "dataset_arrays": data.metadata["arrays"],
        "cases": [
            {"dev_idx": int(data.idx[row]), "truth": data.theta[row].tolist(), "noise_seed": seed}
            for row, seed in zip(row_indices, noise_seeds, strict=True)
        ],
        "reference_summary": reference_summary,
        "npe_sample_summary": {
            "mean": samples.mean(axis=0).tolist(),
            "std": samples.std(axis=0).tolist(),
        },
        "npe_quadrature_integral": npe_integral,
        "grid_comparison": grid_comparison,
        "quadrature_comparison": quadrature_comparison,
        "grid_pass": convergence_pass(grid_comparison, options).tolist(),
        "quadrature_pass": convergence_pass(quadrature_comparison, options).tolist(),
        "npe_vs_reference": comparison,
        "prior_particle_reference": particle_reference,
        "interpolation_validation": {
            "n": len(validation),
            "noise_norm_quantiles_50_90_100": np.quantile(noise_norm, [0.5, 0.9, 1]).tolist(),
            "fraction_below_threshold": float(
                np.mean(noise_norm <= options["max_interpolation_noise_norm"])
            ),
            "log_likelihood_error_at_truth": truth_errors.tolist(),
        },
        "reference_certified": bool(
            np.all(convergence_pass(grid_comparison, options))
            and np.all(convergence_pass(quadrature_comparison, options))
            and np.all(noise_norm <= options["max_interpolation_noise_norm"])
        ),
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "reference.py",
                "reference_diagnostics.py",
                "data.py",
                "model.py",
                "obs_model.py",
                "inference.py",
            )
        },
        "grid_sha256": {
            str(size): file_sha256(grid_directory / f"grid_{size}.npz") for size in grids
        },
        "grid_directory": str(grid_directory.resolve()),
    }
    destination = directory / "analysis"
    destination.mkdir(exist_ok=True)
    np.savez(
        destination / "posteriors.npz",
        a=a,
        c=c,
        reference_mass=reference,
        coarse_reference_mass=masses[min(grids)],
        npe_mass=npe_mass,
        samples=samples,
        observations=observations,
        validation_idx=data.idx[validation],
        interpolation_noise_norm=noise_norm,
    )
    write_json(destination / "summary.json", report)
    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    fig, panels = plt.subplots(
        rows, columns, figsize=(4 * columns, 3.5 * rows), layout="constrained", squeeze=False
    )
    for case, panel in enumerate(panels.flat):
        if case >= len(selected):
            panel.set_axis_off()
            continue
        draw_contours(panel, a, c, reference[case], "#2166ac")
        draw_contours(panel, a, c, npe_mass[case], "#b35806")
        truth = data.theta[selected[case]]
        panel.plot(truth[0], np.cos(np.deg2rad(truth[1])), "k+")
        panel.set(
            xlabel="Spin a",
            ylabel="cos(inclination)",
            title=f"Dev {data.idx[selected[case]]}: a={truth[0]:.3f}, i={truth[1]:.1f}°",
            xlim=(axes[0][0], axes[0][-1]),
            ylim=(axes[1][0], axes[1][-1]),
        )
    status = (
        "convergence checks passed"
        if report["reference_certified"]
        else "reference grid NOT certified"
    )
    fig.suptitle(
        "Approximate 50% / 90% contours in (a, cos i) · blue: likelihood · orange: NPE\n"
        f"Noise seed {options['noise_seeds'][0]} · {status}"
    )
    fig.savefig(destination / "posterior_comparison.png", dpi=options["figure_dpi"])
    plt.close(fig)
    print(
        json.dumps(
            {
                "reference_certified": report["reference_certified"],
                "grid_pass": report["grid_pass"],
                "interpolation_validation": report["interpolation_validation"],
            }
        ),
        flush=True,
    )
    return report
