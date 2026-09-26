import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.radius import (
    PRIOR_NAMES,
    RadiusCache,
    RadiusMesh,
    integrate_radius,
    isco_radius,
    physical_radius,
)
from kerr_sbi.reference import compare_mass, mass_summary, relative_log_likelihood
from kerr_sbi.reference_diagnostics import convergence_pass
from kerr_sbi.reference_mesh import ImageMesh, midpoint_points


def physical_radius_cdf(
    a: np.ndarray, joint_mass: np.ndarray, radius_max: float, query: np.ndarray
) -> np.ndarray:
    t_edges = np.linspace(0, 1, joint_mass.shape[-1] + 1)
    lower = isco_radius(a)
    edges = lower[:, None] + t_edges * (radius_max - lower[:, None])
    cumulative = np.concatenate(
        [np.zeros((*joint_mass.shape[:-1], 1)), np.cumsum(joint_mass, axis=-1)], axis=-1
    )
    rows = cumulative.reshape(-1, len(a), len(t_edges))
    result = np.zeros((len(rows), len(query)))
    for case, row in enumerate(rows):
        for spin, values in enumerate(row):
            result[case] += np.interp(query, edges[spin], values, left=0, right=values[-1])
    return result.reshape(*joint_mass.shape[:-2], len(query))


def summarize_radius(axes: list[np.ndarray], mass: np.ndarray, radius_max: float) -> dict:
    a, c, t = axes
    probability = mass.astype(np.float64)
    probability /= probability.sum(axis=(-3, -2, -1), keepdims=True)
    spin = probability.sum(axis=(-2, -1))
    inclination = probability.sum(axis=(-3, -1))
    joint = probability.sum(axis=-2)
    radii = isco_radius(a)[:, None] + t[None] * (radius_max - isco_radius(a)[:, None])
    means, stds, quantiles, cdfs = [], [], [], []
    values = (a, np.rad2deg(np.arccos(c)), radii.ravel())
    marginals = (spin, inclination, joint.reshape(*joint.shape[:2], -1))
    cdf_values = (a, c, radii.ravel())
    for support, marginal, cdf_support in zip(values, marginals, cdf_values, strict=True):
        mean = np.sum(marginal * support, axis=-1)
        means.append(mean)
        stds.append(np.sqrt(np.maximum(np.sum(marginal * support**2, axis=-1) - mean**2, 0)))
        order = np.argsort(support)
        sorted_mass = marginal[..., order]
        cumulative = np.cumsum(sorted_mass, axis=-1) - sorted_mass / 2
        quantiles.append(
            np.array(
                [
                    np.interp([0.05, 0.5, 0.95], row, support[order])
                    for row in cumulative.reshape(-1, len(support))
                ]
            ).reshape(*marginal.shape[:2], 3)
        )
        order = np.argsort(cdf_support)
        sorted_mass = marginal[..., order]
        cumulative = np.cumsum(sorted_mass, axis=-1) - sorted_mass / 2
        if cdf_support is a or cdf_support is c:
            delta = (cdf_support[1] - cdf_support[0]) / 2
            support_edges = np.r_[cdf_support[0] - delta, cdf_support + delta]
            cumulative = np.concatenate(
                [np.zeros((*marginal.shape[:2], 1)), np.cumsum(sorted_mass, axis=-1)], axis=-1
            )
            grid = np.linspace(support_edges[0], support_edges[-1], 513)
        else:
            grid = np.linspace(float(isco_radius(0.98)), radius_max, 513)
            cdfs.append(physical_radius_cdf(a, joint, radius_max, grid))
            quantile_grid = np.linspace(grid[0], grid[-1], 4097)
            radius_cdf = physical_radius_cdf(a, joint, radius_max, quantile_grid)
            quantiles[-1] = np.array(
                [
                    np.interp([0.05, 0.5, 0.95], row, quantile_grid)
                    for row in radius_cdf.reshape(-1, len(quantile_grid))
                ]
            ).reshape(*marginal.shape[:2], 3)
            continue
        cdfs.append(
            np.array(
                [
                    np.interp(grid, support_edges, row)
                    for row in cumulative.reshape(-1, len(support_edges))
                ]
            ).reshape(*marginal.shape[:2], 513)
        )
    prior = np.ones((3, len(a)))
    prior[2] = radius_max - isco_radius(a)
    prior /= prior.sum(axis=1, keepdims=True)
    prior_mean = np.sum(prior * a, axis=1)
    prior_std = np.sqrt(np.sum(prior * a**2, axis=1) - prior_mean**2)
    kl = np.sum(spin * np.log(np.maximum(spin, np.finfo(float).tiny) / prior[:, None]), axis=-1)
    return {
        "mean": np.stack(means, axis=-1),
        "std": np.stack(stds, axis=-1),
        "quantiles_05_50_95": np.stack(quantiles, axis=-2),
        "cdf": np.stack(cdfs, axis=-2),
        "spin_marginal": spin,
        "spin_radius_fraction_mass": joint,
        "spin_prior": prior,
        "spin_prior_std": prior_std,
        "spin_kl_nats": kl,
        "spin_std_over_prior": np.stack(stds, axis=-1)[..., 0] / prior_std[:, None],
    }


def compare_radius(old: dict, new: dict, options: dict) -> dict:
    comparison = {
        "marginal_cdf_change": np.max(np.abs(old["cdf"] - new["cdf"]), axis=-1),
        "mean_shift_in_std": np.abs(old["mean"] - new["mean"]) / new["std"],
        "relative_std_change": np.abs(old["std"] / new["std"] - 1),
    }
    comparison["passed"] = np.all(
        np.stack([value <= options[f"max_{name}"] for name, value in comparison.items()]),
        axis=(0, 3),
    )
    return {name: value.tolist() for name, value in comparison.items()}


def refinement_points(
    mesh: RadiusMesh, mass: np.ndarray, points: np.ndarray, count: int
) -> np.ndarray:
    simplex, _ = mesh.weights(points)
    score = np.zeros(len(mesh.gram))
    for row in mass.reshape(-1, len(points)):
        score = np.maximum(score, np.bincount(simplex, weights=row, minlength=len(score)))
    faces = mesh.points[mesh.triangulation.convex_hull]
    faces = faces[~np.all(faces[:, :, 2] == 0, axis=1)]
    face_centers = faces.mean(axis=1)
    for dimension in range(3):
        for edge in mesh.bounds[:, dimension]:
            face_centers[np.all(faces[:, :, dimension] == edge, axis=1), dimension] = edge
    face_simplex, _ = mesh.weights(face_centers)
    face_order = np.argsort(-score[face_simplex], kind="stable")[: count // 4]
    remaining = count - len(face_order)
    valid = np.isfinite(mesh.triangulation.transform).all(axis=(1, 2))
    score[~valid] = -np.inf
    order = np.argsort(-score, kind="stable")[:remaining]
    interior = mesh.points[mesh.triangulation.simplices[order]].mean(axis=1)
    return np.clip(np.concatenate([interior, face_centers[face_order]]), *mesh.bounds)


def render_points(
    cache: RadiusCache, points: np.ndarray, radius_max: float, label: str
) -> np.ndarray:
    images = []
    for n, (point, radius) in enumerate(
        zip(points, physical_radius(points, radius_max), strict=True), 1
    ):
        inclination = np.clip(
            np.rad2deg(np.arccos(point[1])),
            cache.cfg["prior"]["incl_min_deg"],
            cache.cfg["prior"]["incl_max_deg"],
        )
        images.append(cache.image(float(point[0]), float(inclination), float(radius)))
        if n % 8 == 0 or n == len(points):
            print(f"{label}: {n}/{len(points)}", flush=True)
    return np.stack(images)


def serializable_summary(summary: dict) -> dict:
    return {
        name: value.tolist()
        for name, value in summary.items()
        if name not in ("cdf", "spin_marginal", "spin_radius_fraction_mass", "spin_prior")
    }


def analyze_radius(
    root: Path, directory: Path, data_directory: Path, cfg: dict, options: dict, cache: RadiusCache
) -> dict:
    with np.load(data_directory / "observations.npz") as archive:
        observations, truth_images, truth = (
            archive["observations"],
            archive["images"],
            archive["truth"],
        )
    cases = json.loads((directory / "cases.json").read_text())["cases"]
    sigma, radius_max = cfg["observation"]["sigma_n"], options["radius_max"]
    fine_shape, coarse_shape = options["quadrature_shapes"][-1], options["quadrature_shapes"][0]
    history, summaries = [], []
    consecutive = np.zeros((3, len(observations)), dtype=int)
    grid_hashes = {}

    def evaluate(
        mesh: RadiusMesh, label: str
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict]:
        nonlocal consecutive
        print(f"Integrating {label}: {len(mesh.points)} nodes, quadrature {fine_shape}", flush=True)
        axes, mass, points = integrate_radius(mesh, observations, sigma, radius_max, fine_shape)
        summary = summarize_radius(axes, mass, radius_max)
        entry = {"label": label, "nodes": len(mesh.points)}
        if summaries:
            comparison = compare_radius(summaries[-1], summary, options)
            entry["comparison"] = comparison
            passed = np.array(comparison["passed"])
            consecutive = np.where(passed, consecutive + 1, 0)
            print(f"{label}: {passed.sum()}/{passed.size} prior/case refinements pass", flush=True)
        history.append(entry)
        summaries.append(summary)
        write_json(directory / "status.json", {"stage": label, "history": history})
        return axes, mass, points, summary

    for size in options["grid_sizes"]:
        path = data_directory / f"grid_{size}.npz"
        grid_hashes[str(size)] = file_sha256(path)
        with np.load(path) as archive:
            mesh_points, mesh_images, bounds = (
                archive["points"],
                archive["images"],
                archive["bounds"],
            )
        mesh = RadiusMesh(mesh_points, mesh_images, bounds)
        axes, mass, quadrature, summary = evaluate(mesh, f"grid_{size}")
    for n, count in enumerate(options["refinement_counts"], 1):
        added = refinement_points(mesh, mass, quadrature, count)
        added_images = render_points(cache, added, radius_max, f"Refinement {n}")
        mesh_points = np.concatenate([mesh_points, added])
        mesh_images = np.concatenate([mesh_images, added_images])
        mesh = RadiusMesh(mesh_points, mesh_images, bounds)
        axes, mass, quadrature, summary = evaluate(mesh, f"refinement_{n}")
        np.savez(
            data_directory / f"refinement_{n}.npz",
            points=mesh_points,
            images=mesh_images,
            bounds=bounds,
        )
        if np.all(consecutive >= options["required_successive_passes"]):
            break
    coarse_axes, coarse_mass, _ = integrate_radius(
        mesh, observations, sigma, radius_max, coarse_shape
    )
    quadrature_comparison = compare_radius(
        summarize_radius(coarse_axes, coarse_mass, radius_max), summary, options
    )
    del coarse_mass
    rng = np.random.default_rng(options["confirmation_seed"])
    confirmation = []
    for case in range(len(cases)):
        mixture = mass[:, case].mean(axis=0, dtype=np.float64).ravel()
        mixture /= mixture.sum()
        chosen = rng.choice(len(quadrature), options["confirmation_per_case"], p=mixture)
        confirmation.extend(quadrature[chosen])
    confirmation = np.array(confirmation)
    confirmation += rng.uniform(-0.5, 0.5, confirmation.shape) * (
        (bounds[1] - bounds[0]) / fine_shape
    )
    direct = render_points(cache, confirmation, radius_max, "Posterior confirmation")
    prediction = mesh.interpolate(confirmation)
    noise_norm = np.linalg.norm((prediction - direct).reshape(len(direct), -1), axis=1) / sigma
    error = relative_log_likelihood(observations, prediction, sigma) - relative_log_likelihood(
        observations, direct, sigma
    )
    confirmation_cases = np.repeat(np.arange(len(cases)), options["confirmation_per_case"])
    boundary = np.array(
        [
            [a, np.cos(np.deg2rad(i)), t]
            for a, i in options["pairs"]
            for t in options["boundary_fractions"]
        ]
    )
    boundary_images = render_points(cache, boundary, radius_max, "Boundary confirmation")
    boundary_norm = (
        np.linalg.norm(
            (mesh.interpolate(boundary) - boundary_images).reshape(len(boundary), -1), axis=1
        )
        / sigma
    )
    truth_norm = (
        np.linalg.norm((mesh.interpolate(truth) - truth_images).reshape(len(truth), -1), axis=1)
        / sigma
    )
    case_validation = (
        noise_norm.reshape(len(cases), -1).max(axis=1) <= options["max_interpolation_noise_norm"]
    )
    case_validation &= (
        truth_norm[np.array([case["truth_index"] for case in cases])]
        <= options["max_interpolation_noise_norm"]
    )
    boundary_pass = bool(np.all(boundary_norm <= options["max_interpolation_noise_norm"]))
    certified = (
        (consecutive >= options["required_successive_passes"])
        & np.array(quadrature_comparison["passed"])
        & case_validation[None]
        & boundary_pass
    )
    source_data = root / "data/reference" / options["source_run"]
    source_report = json.loads((root / "runs" / options["source_run"] / "summary.json").read_text())
    with np.load(source_data / "mesh.npz") as archive:
        isco_points, isco_images, isco_bounds = (
            archive["points"],
            archive["images"],
            archive["bounds"],
        )
    previous_counts = [entry["nodes"] for entry in source_report["history"][-3:-1]]
    if len(previous_counts) != 2:
        raise ValueError("ISCO source needs two saved forward refinements")
    isco_mesh = ImageMesh(isco_points, isco_images, isco_bounds)
    a, c, isco_mass = isco_mesh.mass(observations, sigma, 512)
    previous_masses = [
        ImageMesh(isco_points[:count], isco_images[:count], isco_bounds).mass(
            observations, sigma, 512
        )[2]
        for count in previous_counts
    ] + [isco_mass]
    isco_grid_checks = [
        compare_mass(a, c, old, new)
        for old, new in zip(previous_masses[:-1], previous_masses[1:], strict=True)
    ]
    isco_grid = isco_grid_checks[-1]
    a0, c0, old_mass = isco_mesh.mass(observations, sigma, 256)
    isco_quad = compare_mass(
        a0, c0, old_mass, isco_mass.reshape(-1, 256, 2, 256, 2).sum(axis=(2, 4))
    )
    _, _, isco_quadrature = midpoint_points(isco_bounds, 512)
    isco_confirmation = np.stack(
        [isco_quadrature[rng.choice(len(isco_quadrature), p=row.ravel())] for row in isco_mass]
    )
    isco_confirmation += rng.uniform(-0.5, 0.5, isco_confirmation.shape) * (
        (isco_bounds[1] - isco_bounds[0]) / 512
    )
    isco_direct = render_points(
        cache,
        np.column_stack([isco_confirmation, np.zeros(len(cases))]),
        radius_max,
        "ISCO reference confirmation",
    )
    isco_norm = (
        np.linalg.norm(
            (isco_mesh.interpolate(isco_confirmation) - isco_direct).reshape(len(cases), -1), axis=1
        )
        / sigma
    )
    isco_certified = (
        np.all([convergence_pass(check, options) for check in isco_grid_checks], axis=0)
        & convergence_pass(isco_quad, options)
        & (isco_norm <= options["max_interpolation_noise_norm"])
    )
    isco_summary = mass_summary(a, c, isco_mass)
    isco_spin = isco_mass.sum(axis=2)
    isco_kl = np.sum(
        isco_spin * np.log(np.maximum(isco_spin, np.finfo(float).tiny) * len(a)), axis=1
    )
    prior_cdf_difference = np.max(
        np.abs(summary["cdf"][0, :, 0] - summary["cdf"][1, :, 0]), axis=-1
    )
    report = {
        "settings": options,
        "library_versions": {name: version(name) for name in ("numpy", "scipy", "jax")},
        "experiment_sha256": file_sha256(directory / "experiment.json"),
        "observations_sha256": file_sha256(data_directory / "observations.npz"),
        "cases": cases,
        "prior_names": list(PRIOR_NAMES),
        "coordinate_order": ["spin", "cos_inclination", "radius_fraction"],
        "unused_singular_tetrahedra": int(
            np.sum(~np.isfinite(mesh.triangulation.transform).all(axis=(1, 2)))
        ),
        "reported_moment_order": ["spin", "inclination_deg", "physical_radius"],
        "posterior_archive_axes": ["prior", "case", "spin", "cos_inclination", "radius_fraction"],
        "radius_transform": (
            "r = ISCO(a) + t * (radius_max - ISCO(a)); saved mass is cell probability"
        ),
        "cdf_coordinate_order": ["spin", "cos_inclination", "physical_radius"],
        "priors": {
            "uniform_radius": "Uniform a,c; p(r|a)=1/(8-ISCO(a))",
            "log_uniform_radius": "Uniform a,c; p(r|a)=1/(r log(8/ISCO(a)))",
            "flat_joint": "Constant density in allowed (a,c,r); p(a) proportional to 8-ISCO(a)",
        },
        "history": history,
        "quadrature_comparison": quadrature_comparison,
        "consecutive_refinement_passes": consecutive.tolist(),
        "certified_by_prior_and_case": certified.tolist(),
        "all_certified": bool(np.all(certified)),
        "summary": serializable_summary(summary),
        "uniform_vs_log_radius_spin_cdf_difference": prior_cdf_difference.tolist(),
        "validation": {
            "posterior_noise_norm": noise_norm.tolist(),
            "posterior_cases": confirmation_cases.tolist(),
            "posterior_log_likelihood_error": error[
                confirmation_cases, np.arange(len(direct))
            ].tolist(),
            "boundary_points": boundary.tolist(),
            "boundary_noise_norm": boundary_norm.tolist(),
            "truth_noise_norm": truth_norm.tolist(),
            "boundary_pass": boundary_pass,
        },
        "isco_reference": {
            "summary": isco_summary,
            "spin_kl_nats": isco_kl.tolist(),
            "grid_comparison": isco_grid,
            "grid_comparisons": isco_grid_checks,
            "quadrature_comparison": isco_quad,
            "confirmation_noise_norm": isco_norm.tolist(),
            "certified": isco_certified.tolist(),
            "interpretation": "ISCO fit to radius=8 truths is a model-misspecification check",
        },
        "grid_sha256": grid_hashes,
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (
                root / "src/kerr_sbi/radius.py",
                Path(__file__).resolve(),
                root / "scripts/07_radius.py",
                root / "src/kerr_sbi/reference_mesh.py",
                root / "src/kerr_sbi/obs_model.py",
            )
        },
        "certification_scope": (
            "Finite numerical checks for this pilot; unresolved cases are exploratory"
        ),
    }
    np.savez(data_directory / "mesh.npz", points=mesh_points, images=mesh_images, bounds=bounds)
    report["mesh_sha256"] = file_sha256(data_directory / "mesh.npz")
    np.savez(
        directory / "posteriors.npz",
        a=axes[0],
        c=axes[1],
        t=axes[2],
        posterior_mass=mass,
        spin_mass=summary["spin_marginal"],
        spin_radius_fraction_mass=summary["spin_radius_fraction_mass"],
        spin_prior=summary["spin_prior"],
        isco_a=a,
        isco_c=c,
        isco_mass=isco_mass,
        confirmation_points=confirmation,
        confirmation_cases=confirmation_cases,
        isco_confirmation_points=isco_confirmation,
        cdf=summary["cdf"],
    )
    report["posterior_archive_sha256"] = file_sha256(directory / "posteriors.npz")
    write_json(directory / "summary.json", report)
    plot_radius(directory, options, cases, axes, summary, certified, a, isco_spin, isco_certified)
    write_json(
        directory / "status.json",
        {
            "stage": "complete",
            "all_certified": bool(np.all(certified)),
            "certified_count": int(certified.sum()),
        },
    )
    return report


def plot_radius(
    directory: Path,
    options: dict,
    cases: list[dict],
    axes: list[np.ndarray],
    summary: dict,
    certified: np.ndarray,
    isco_a: np.ndarray,
    isco_spin: np.ndarray,
    isco_certified: np.ndarray,
) -> None:
    a, _, t = axes
    count = 2 * len(options["pairs"])
    colors = ["#2166ac", "#b35806", "#7570b3"]
    prior_labels = ("Uniform radius", "Log-uniform radius", "Flat joint")
    fig, panels = plt.subplots(
        len(options["pairs"]), 2, figsize=(11, 3 * len(options["pairs"])), layout="constrained"
    )
    for case, panel in enumerate(panels.flat):
        for prior, name in enumerate(prior_labels):
            panel.plot(
                a,
                summary["spin_marginal"][prior, case] / (a[1] - a[0]),
                color=colors[prior],
                label=name,
            )
        panel.plot(
            isco_a,
            isco_spin[case] / (isco_a[1] - isco_a[0]),
            color="#555555",
            linestyle="--",
            label="ISCO model",
        )
        panel.axvline(cases[case]["spin"], color="black", linewidth=0.8)
        status = (
            "checks pass"
            if np.all(certified[:, case]) and isco_certified[case]
            else "NUMERICALLY UNRESOLVED"
        )
        panel.set(
            xlabel="Spin a",
            ylabel="Marginal density",
            title=(
                f"a={cases[case]['spin']:.2f}, i={cases[case]['inclination_deg']:.0f}°, "
                f"r={cases[case]['radius']:.2f}M\n{status}"
            ),
        )
        if case == 0:
            panel.legend(fontsize=8)
    fig.suptitle(f"Spin marginals · noise seed {options['noise_seeds'][0]} · radius marginalized")
    fig.savefig(directory / "spin_marginals.png", dpi=options["figure_dpi"])
    plt.close(fig)
    fig, panels = plt.subplots(
        len(options["pairs"]), 2, figsize=(11, 3 * len(options["pairs"])), layout="constrained"
    )
    a_edges = np.linspace(a[0] - (a[1] - a[0]) / 2, a[-1] + (a[1] - a[0]) / 2, len(a) + 1)
    t_edges = np.linspace(0, 1, len(t) + 1)
    lower = isco_radius(np.clip(a_edges, 0, 0.98))
    radii = lower[:, None] + t_edges[None] * (options["radius_max"] - lower[:, None])
    for case, panel in enumerate(panels.flat):
        density = summary["spin_radius_fraction_mass"][0, case] / (
            (a[1] - a[0]) * (t[1] - t[0]) * (options["radius_max"] - isco_radius(a))[:, None]
        )
        artist = panel.pcolormesh(
            np.broadcast_to(a_edges, radii.T.shape),
            radii.T,
            density.T,
            shading="flat",
            cmap="magma",
        )
        fig.colorbar(artist, ax=panel, label="Joint density / M⁻¹")
        panel.plot(a_edges, lower, color="white", linewidth=0.7)
        panel.plot(cases[case]["spin"], cases[case]["radius"], "+", color="#00ffff")
        status = "checks pass" if certified[0, case] else "NUMERICALLY UNRESOLVED"
        panel.set(
            xlabel="Spin a",
            ylabel="Inner radius / M",
            title=(
                f"a={cases[case]['spin']:.2f}, i={cases[case]['inclination_deg']:.0f}°, "
                f"r={cases[case]['radius']:.2f}M\n{status}"
            ),
        )
    fig.suptitle(
        f"Spin–radius density · noise seed {options['noise_seeds'][0]}\n"
        "Inclination marginalized · conditional uniform-radius prior"
    )
    fig.savefig(directory / "spin_radius_joint.png", dpi=options["figure_dpi"])
    plt.close(fig)
    fig, panels = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for prior, name in enumerate(prior_labels):
        for seed in range(len(options["noise_seeds"])):
            selection = slice(seed * count, (seed + 1) * count)
            x = np.arange(count) + (prior - 1) * 0.2
            panels[0].scatter(
                x,
                summary["spin_std_over_prior"][prior, selection],
                edgecolors=colors[prior],
                facecolors=[
                    colors[prior] if value else "none" for value in certified[prior, selection]
                ],
                marker=("o", "s", "^")[seed],
                label=name if seed == 0 else None,
            )
            panels[1].scatter(
                x,
                summary["spin_kl_nats"][prior, selection],
                edgecolors=colors[prior],
                facecolors=[
                    colors[prior] if value else "none" for value in certified[prior, selection]
                ],
                marker=("o", "s", "^")[seed],
            )
    labels = [
        f"{case['spin']:.2f}, {case['inclination_deg']:.0f}°\n"
        f"{'ISCO' if case['fraction'] == 0 else '8M'}"
        for case in cases[:count]
    ]
    for panel, label in zip(
        panels, ("Posterior spin std / its prior std", "Spin information gain / nats"), strict=True
    ):
        panel.set(xticks=np.arange(count), xticklabels=labels, ylabel=label)
        panel.tick_params(axis="x", labelsize=8)
    panels[0].axhline(1, color="grey", linestyle=":")
    panels[0].legend(fontsize=8)
    fig.suptitle("All three noise seeds · filled: numerical checks pass · open: unresolved")
    fig.savefig(directory / "spin_constraints.png", dpi=options["figure_dpi"])
    plt.close(fig)

    fig, panels = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    panels[0].plot(
        a,
        summary["spin_prior"][0] / (a[1] - a[0]),
        color="black",
        label="Both conditional radius priors",
    )
    panels[0].plot(
        a, summary["spin_prior"][2] / (a[1] - a[0]), color=colors[2], label="Flat joint prior"
    )
    panels[0].set(xlabel="Spin a", ylabel="Prior density", title="Induced spin prior")
    panels[0].legend(fontsize=8)
    for spin in sorted({pair[0] for pair in options["pairs"]}):
        lower = float(isco_radius(spin))
        radius = np.linspace(lower, options["radius_max"], 200)
        line = panels[1].plot(
            radius,
            np.full_like(radius, 1 / (options["radius_max"] - lower)),
            label=f"a={spin:g}: uniform radius",
        )[0]
        panels[1].plot(
            radius,
            1 / (radius * np.log(options["radius_max"] / lower)),
            linestyle="--",
            color=line.get_color(),
            label=f"a={spin:g}: log-uniform",
        )
    panels[1].set(
        xlabel="Inner radius / M",
        ylabel="Conditional prior density",
        title="Radius priors at fixed spin",
    )
    panels[1].legend(fontsize=8)
    fig.savefig(directory / "radius_priors.png", dpi=options["figure_dpi"])
    plt.close(fig)
