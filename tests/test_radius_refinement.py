import numpy as np
import pytest

from kerr_sbi.radius import RadiusMesh, integrate_radius
from kerr_sbi.radius_diagnostics import summarize_radius, summarize_radius_marginals
from kerr_sbi.radius_refinement import (
    draw_radius_checks,
    integrate_radius_slabs,
    targeted_radius_points,
)


def mesh_fixture():
    bounds = np.array([[0.0, 0.2, 0.0], [0.98, 1.0, 1.0]])
    points = np.stack(
        np.meshgrid(*[np.linspace(*bounds[:, j], 3) for j in range(3)], indexing="ij"), -1
    ).reshape(-1, 3)
    return RadiusMesh(points, points, bounds)


def test_slab_integral_matches_full_mass_and_simplex_scores():
    mesh = mesh_fixture()
    observation = np.array([0.7, 0.6, 0.3])
    axes, mass, points = integrate_radius(mesh, observation[None], 0.15, 8, [32, 24, 20])
    expected = summarize_radius(axes, mass, 8)
    result = integrate_radius_slabs(mesh, observation, 0.15, 8, [32, 24, 20], True)
    actual = summarize_radius_marginals(
        result["axes"], result["spin"], result["inclination"], result["joint"], 8
    )
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-8)
    simplex, _ = mesh.weights(points)
    expected_scores = np.stack(
        [np.bincount(simplex, weights=row.ravel(), minlength=len(mesh.gram)) for row in mass[:, 0]]
    )
    np.testing.assert_allclose(result["simplex_scores"], expected_scores, atol=2e-8)
    np.testing.assert_allclose(result["simplex_scores"].sum(axis=1), 1, atol=1e-12)


def test_slab_gaussian_moments_and_mixture_draws():
    mesh = mesh_fixture()
    observation = np.array([0.49, 0.6, 0.5])
    result = integrate_radius_slabs(mesh, observation, 0.05, 8, [48, 48, 48])
    summary = summarize_radius_marginals(
        result["axes"], result["spin"], result["inclination"], result["joint"], 8
    )
    assert summary["mean"][0, 0, 0] == pytest.approx(0.49, abs=1e-9)
    assert summary["std"][0, 0, 0] == pytest.approx(0.05, abs=1e-8)
    draws = draw_radius_checks(mesh, observation, 0.05, result, 2500, np.random.default_rng(7))
    assert np.all(draws >= mesh.bounds[0]) and np.all(draws <= mesh.bounds[1])
    assert draws[:, 0].mean() == pytest.approx(summary["mean"][:, 0, 0].mean(), abs=0.004)
    assert draws[:, 1].mean() == pytest.approx(0.6, abs=0.004)
    assert draws[:, 2].mean() == pytest.approx(0.5, abs=0.004)


def test_error_targeting_includes_isco_face_and_excludes_existing_points():
    mesh = mesh_fixture()
    diagnostic = np.array([[0.8, 0.7, 0.1], [0.4, 0.4, 0.4]])
    candidates, reasons = targeted_radius_points(
        mesh, np.ones(len(mesh.gram)), diagnostic, np.array([1.4, 0.01]), 16
    )
    assert np.any(np.all(candidates == diagnostic[0], axis=1))
    assert np.any(candidates[:, 2] == 0)
    assert "isco_face_mass" in reasons
    assert (
        len(np.unique(np.concatenate([mesh.points, candidates]), axis=0)) == len(mesh.points) + 16
    )
    with pytest.raises(ValueError, match="diagnostic errors"):
        targeted_radius_points(mesh, np.ones(len(mesh.gram)), diagnostic, np.array([np.nan, 1]), 16)
