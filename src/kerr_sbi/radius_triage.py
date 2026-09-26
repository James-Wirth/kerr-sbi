import numpy as np

from kerr_sbi.radius import RadiusMesh

METRICS = ("marginal_cdf_change", "mean_shift_in_std", "relative_std_change")


def comparison_pass(comparison: dict, options: dict) -> np.ndarray:
    values = [np.asarray(comparison[name], dtype=float) for name in METRICS]
    if any(not np.isfinite(value).all() or np.any(value < 0) for value in values):
        raise ValueError("comparison errors must be finite and nonnegative")
    return np.logical_and.reduce(
        [
            np.all(value <= options[f"max_{name}"], axis=-1)
            for name, value in zip(METRICS, values, strict=True)
        ]
    )


def failure_table(report: dict) -> list[dict]:
    options, cases = report["settings"], report["cases"]
    shape = (len(report["prior_names"]), len(cases))
    consecutive = np.zeros(shape, dtype=int)
    history = [entry for entry in report["history"] if "comparison" in entry]
    for entry in history:
        passed = comparison_pass(entry["comparison"], options)
        if not np.array_equal(passed, entry["comparison"]["passed"]):
            raise ValueError("saved forward comparison flags differ")
        consecutive = np.where(passed, consecutive + 1, 0)
    if not np.array_equal(consecutive, report["consecutive_refinement_passes"]):
        raise ValueError("saved consecutive refinement counts differ")
    quadrature = comparison_pass(report["quadrature_comparison"], options)
    if not np.array_equal(quadrature, report["quadrature_comparison"]["passed"]):
        raise ValueError("saved quadrature flags differ")
    validation, isco = report["validation"], report["isco_reference"]
    threshold = options["max_interpolation_noise_norm"]
    for errors in (
        validation["posterior_noise_norm"],
        validation["boundary_noise_norm"],
        validation["truth_noise_norm"],
        isco["confirmation_noise_norm"],
    ):
        values = np.asarray(errors)
        if not values.size or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("interpolation errors must be finite and nonnegative")
    posterior_cases = np.asarray(validation["posterior_cases"])
    posterior_errors = np.asarray(validation["posterior_noise_norm"])
    if (
        posterior_cases.shape != posterior_errors.shape
        or not np.isin(posterior_cases, np.arange(len(cases))).all()
    ):
        raise ValueError("invalid posterior case assignments")
    boundary_error = float(np.max(validation["boundary_noise_norm"]))
    isco_forward = np.logical_and.reduce(
        [comparison_pass(check, options) for check in isco["grid_comparisons"]]
    )
    isco_quad = comparison_pass(isco["quadrature_comparison"], options)
    rows, accepted, isco_accepted = [], np.zeros(shape, dtype=bool), []
    for case_index, case in enumerate(cases):
        selected = posterior_errors[posterior_cases == case_index]
        if not len(selected):
            raise ValueError("missing posterior validation for case")
        posterior_error = float(selected.max())
        truth_error = float(validation["truth_noise_norm"][case["truth_index"]])
        isco_error = float(isco["confirmation_noise_norm"][case_index])
        isco_ok = bool(
            isco_forward[case_index] and isco_quad[case_index] and isco_error <= threshold
        )
        isco_accepted.append(isco_ok)
        for prior_index, prior in enumerate(report["prior_names"]):
            row = {"case": case_index, "prior": prior, **case}
            checks = {
                "forward": consecutive[prior_index, case_index]
                >= options["required_successive_passes"],
                "quadrature": quadrature[prior_index, case_index],
                "truth": truth_error <= threshold,
                "posterior": posterior_error <= threshold,
                "boundary": boundary_error <= threshold,
            }
            row.update({f"{name}_pass": bool(value) for name, value in checks.items()})
            row.update(
                forward_consecutive=int(consecutive[prior_index, case_index]),
                forward_required=options["required_successive_passes"],
                truth_noise_norm=truth_error,
                posterior_noise_norm=posterior_error,
                boundary_max_noise_norm=boundary_error,
                interpolation_threshold=threshold,
                isco_forward_pass=bool(isco_forward[case_index]),
                isco_quadrature_pass=bool(isco_quad[case_index]),
                isco_noise_norm=isco_error,
                isco_accepted=isco_ok,
                accepted=bool(all(checks.values())),
                failed_checks=";".join(name for name, passed in checks.items() if not passed),
            )
            for name in METRICS:
                row[f"{name}_threshold"] = options[f"max_{name}"]
                for entry in history:
                    row[f"{entry['label']}_{name}_max"] = float(
                        np.max(entry["comparison"][name][prior_index][case_index])
                    )
                row[f"quadrature_{name}_max"] = float(
                    np.max(report["quadrature_comparison"][name][prior_index][case_index])
                )
                row[f"isco_forward_{name}_max"] = float(
                    max(np.max(check[name][case_index]) for check in isco["grid_comparisons"])
                )
                row[f"isco_quadrature_{name}_max"] = float(
                    np.max(isco["quadrature_comparison"][name][case_index])
                )
            accepted[prior_index, case_index] = row["accepted"]
            rows.append(row)
    if not np.array_equal(accepted, report["certified_by_prior_and_case"]):
        raise ValueError("saved free-radius acceptance flags differ")
    if not np.array_equal(isco_accepted, isco["certified"]):
        raise ValueError("saved ISCO acceptance flags differ")
    if (boundary_error <= threshold) != validation["boundary_pass"]:
        raise ValueError("saved boundary flag differs")
    return rows


def residual_statistics(residual: np.ndarray) -> dict:
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or not np.isfinite(residual).all():
        raise ValueError("finite two-dimensional residual required")
    energy = np.sort(residual.ravel() ** 2)[::-1]
    total = energy.sum()
    return {
        "norm": float(np.sqrt(total)),
        "max_absolute": float(np.max(np.abs(residual))),
        "top_1_percent_energy_fraction": float(
            energy[: max(1, int(np.ceil(0.01 * len(energy))))].sum() / total
        )
        if total
        else 0.0,
        "top_5_percent_energy_fraction": float(
            energy[: max(1, int(np.ceil(0.05 * len(energy))))].sum() / total
        )
        if total
        else 0.0,
    }


def simplex_diagnostics(mesh: RadiusMesh, points: np.ndarray) -> dict[str, np.ndarray]:
    simplex, weights = mesh.weights(points)
    vertices = mesh.points[mesh.triangulation.simplices[simplex]]
    unit_vertices = (vertices - mesh.bounds[0]) / mesh.width
    edges = unit_vertices[:, 1:] - unit_vertices[:, :1]
    unit_points = (points - mesh.bounds[0]) / mesh.width
    return {
        "simplex": simplex,
        "weights": weights,
        "unit_volume": np.abs(np.linalg.det(edges)) / 6,
        "edge_matrix_condition": np.linalg.cond(edges),
        "unit_diameter": np.linalg.norm(
            unit_vertices[:, :, None] - unit_vertices[:, None, :], axis=-1
        ).max(axis=(1, 2)),
        "unit_boundary_distance": np.minimum(unit_points, 1 - unit_points).min(axis=1),
        "radius_fraction": points[:, 2],
    }
