from copy import deepcopy

import numpy as np
import pytest

from kerr_sbi.radius import RadiusMesh
from kerr_sbi.radius_triage import (
    METRICS,
    comparison_pass,
    failure_table,
    residual_statistics,
    simplex_diagnostics,
)


def passing_report() -> dict:
    comparison = {name: [[[0.01, 0.01, 0.01]]] for name in METRICS}
    comparison["passed"] = [[True]]
    isco_comparison = {name: [[0.01, 0.01]] for name in METRICS}
    return {
        "settings": {
            **{f"max_{name}": 0.02 for name in METRICS},
            "required_successive_passes": 2,
            "max_interpolation_noise_norm": 0.25,
        },
        "cases": [{"truth_index": 0}],
        "prior_names": ["uniform_radius"],
        "history": [
            {"label": "first", "comparison": deepcopy(comparison)},
            {"label": "second", "comparison": deepcopy(comparison)},
        ],
        "consecutive_refinement_passes": [[2]],
        "quadrature_comparison": deepcopy(comparison),
        "certified_by_prior_and_case": [[True]],
        "validation": {
            "posterior_cases": [0, 0],
            "posterior_noise_norm": [0.1, 0.25],
            "boundary_noise_norm": [0.01],
            "truth_noise_norm": [0.01],
            "boundary_pass": True,
        },
        "isco_reference": {
            "grid_comparisons": [isco_comparison, isco_comparison],
            "quadrature_comparison": isco_comparison,
            "confirmation_noise_norm": [0.1],
            "certified": [True],
        },
    }


def test_failure_table_reconstructs_flags_and_requires_successive_passes() -> None:
    report = passing_report()
    assert failure_table(report)[0]["accepted"]
    report["history"][0]["comparison"]["marginal_cdf_change"][0][0][0] = 0.021
    report["history"][0]["comparison"]["passed"] = [[False]]
    report["consecutive_refinement_passes"] = [[1]]
    report["certified_by_prior_and_case"] = [[False]]
    row = failure_table(report)[0]
    assert row["failed_checks"] == "forward"
    assert row["forward_consecutive"] == 1
    report["certified_by_prior_and_case"] = [[True]]
    with pytest.raises(ValueError, match="acceptance flags"):
        failure_table(report)


def test_isco_failure_is_separate_from_free_radius_acceptance() -> None:
    report = passing_report()
    report["isco_reference"]["confirmation_noise_norm"] = [0.251]
    report["isco_reference"]["certified"] = [False]
    row = failure_table(report)[0]
    assert row["accepted"] and not row["isco_accepted"]
    assert row["isco_forward_pass"] and row["isco_quadrature_pass"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.01])
def test_invalid_comparison_cannot_be_accepted(invalid: float) -> None:
    report = passing_report()
    report["quadrature_comparison"]["mean_shift_in_std"][0][0][1] = invalid
    with pytest.raises(ValueError, match="finite and nonnegative"):
        comparison_pass(report["quadrature_comparison"], report["settings"])
    report = passing_report()
    report["validation"]["posterior_noise_norm"][0] = invalid
    with pytest.raises(ValueError, match="finite and nonnegative"):
        failure_table(report)


def test_residual_concentration_distinguishes_local_from_diffuse_error() -> None:
    localized = np.zeros((10, 10))
    localized[4, 7] = -1
    local, diffuse = residual_statistics(localized), residual_statistics(np.full((10, 10), 0.1))
    assert local["norm"] == pytest.approx(diffuse["norm"])
    assert local["top_1_percent_energy_fraction"] == 1
    assert diffuse["top_1_percent_energy_fraction"] == pytest.approx(0.01)
    assert residual_statistics(np.zeros((10, 10)))["top_1_percent_energy_fraction"] == 0


def test_simplex_geometry_uses_normalized_coordinates() -> None:
    bounds = np.array([[0, 0.2, 0], [0.98, 1, 1]])
    points = np.array([[0, 0.2, 0], [0.98, 0.2, 0], [0, 1, 0], [0, 0.2, 1]])
    mesh = RadiusMesh(points, np.zeros((4, 1)), bounds)
    geometry = simplex_diagnostics(mesh, points.mean(axis=0, keepdims=True))
    np.testing.assert_allclose(geometry["weights"], 0.25)
    assert geometry["unit_volume"][0] == pytest.approx(1 / 6)
    assert geometry["unit_diameter"][0] == pytest.approx(np.sqrt(2))
    assert geometry["unit_boundary_distance"][0] == pytest.approx(0.25)
