from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kerr_sbi.persistence import write_json
from kerr_sbi.radius import (
    RadiusCache,
    RadiusMesh,
    integrate_radius,
    isco_radius,
    physical_radius,
    radius_prior_weights,
)
from kerr_sbi.radius_diagnostics import (
    analyze_radius,
    compare_radius,
    refinement_points,
    summarize_radius,
)
from kerr_sbi.reference import relative_log_likelihood


def box_points(size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.array([[0.0, 0.2, 0.0], [0.98, 1.0, 1.0]])
    axes = [np.linspace(bounds[0, j], bounds[1, j], size) for j in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3), bounds


def test_isco_matches_circular_orbit_equation_and_known_limits() -> None:
    spin = np.linspace(0, 0.98, 100)
    radius = isco_radius(spin)
    np.testing.assert_allclose(
        radius**2 - 6 * radius + 8 * spin * np.sqrt(radius) - 3 * spin**2, 0, atol=2e-13
    )
    assert isco_radius(0) == 6
    assert isco_radius(0.5) == pytest.approx(4.233002529530826, abs=1e-13)
    assert np.all(np.diff(radius) < 0)
    for value in (-0.1, 0.99, np.nan):
        with pytest.raises(ValueError, match="spin"):
            isco_radius(value)


def test_radius_priors_include_jacobians_and_preserve_intended_spin_prior() -> None:
    spin = np.array([0.0, 0.5, 0.98])
    t = (np.arange(8192) + 0.5) / 8192
    points = np.array([[a, 0.5, fraction] for a in spin for fraction in t])
    weights = radius_prior_weights(points, 8).reshape(3, 3, -1)
    np.testing.assert_allclose(weights[:2].mean(axis=2), 1, atol=1e-8)
    np.testing.assert_allclose(weights[2].mean(axis=1), 8 - isco_radius(spin))
    endpoints = np.array([[a, 0.5, fraction] for a in spin for fraction in (0, 1)])
    np.testing.assert_allclose(physical_radius(endpoints, 8).reshape(3, 2)[:, 0], isco_radius(spin))
    np.testing.assert_allclose(physical_radius(endpoints, 8).reshape(3, 2)[:, 1], 8)
    with pytest.raises(ValueError, match="fraction"):
        physical_radius(np.array([[0.5, 0.5, 1.1]]), 8)


def test_radius_cache_distinguishes_edges_and_rejects_corruption(
    tmp_path: Path, cfg: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_render(a, inclination, stem, config):
        calls.append((a, inclination, config["scene"]["r_in"]))
        path = stem.with_suffix(".pfm")
        path.write_bytes(b"image")
        return path

    monkeypatch.setattr("kerr_sbi.reference.simulator_provenance", lambda cfg: {"version": "test"})
    monkeypatch.setattr("kerr_sbi.radius.render", fake_render)
    monkeypatch.setattr("kerr_sbi.radius.preprocess_pfm", lambda path, cfg: np.ones((64, 64)))
    cache = RadiusCache(tmp_path, cfg)
    cache.image(0.5, 45, 6)
    cache.image(0.5, 45, 8)
    cache.image(0.5, 45, 6)
    assert calls == [(0.5, 45, 6), (0.5, 45, 8)]
    assert cfg["scene"]["r_in"] == 0
    with pytest.raises(ValueError, match="physical radius"):
        cache.image(0.5, 45, 2)
    for path in tmp_path.glob("*.pfm"):
        path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        cache.image(0.5, 45, 6)
    changed = deepcopy(cfg)
    changed["scene"]["mass"] = 2
    with pytest.raises(ValueError, match="mass"):
        RadiusCache(tmp_path, changed)


def test_radius_mesh_matches_direct_likelihood_and_enforces_bounds() -> None:
    points, bounds = box_points(3)
    rng = np.random.default_rng(40)
    images = rng.uniform(0, 100, (len(points), 3, 4))
    mesh = RadiusMesh(points, images, bounds)
    query = bounds[0] + rng.uniform(size=(127, 3)) * (bounds[1] - bounds[0])
    observations = rng.normal(size=(3, 3, 4))
    expected = relative_log_likelihood(observations, mesh.interpolate(query), 20)
    np.testing.assert_allclose(mesh.log_likelihood(query, observations, 20), expected, atol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        mesh.interpolate(np.array([[0.5, 0.5, 1.01]]))
    with pytest.raises(ValueError, match="positive"):
        mesh.log_likelihood(query, observations, 0)


def test_uniform_likelihood_recovers_all_three_spin_priors() -> None:
    points, bounds = box_points()
    mesh = RadiusMesh(points, np.zeros((len(points), 2)), bounds)
    axes, mass, query = integrate_radius(mesh, np.zeros((1, 2)), 1, 8, [64, 16, 64])
    summary = summarize_radius(axes, mass, 8)
    np.testing.assert_allclose(summary["spin_marginal"][:, 0], summary["spin_prior"], atol=2e-6)
    np.testing.assert_allclose(summary["spin_std_over_prior"], 1, atol=2e-5)
    np.testing.assert_allclose(summary["spin_kl_nats"], 0, atol=1e-8)
    assert summary["mean"][2, 0, 0] > summary["mean"][0, 0, 0]
    assert summary["mean"][1, 0, 2] < summary["mean"][0, 0, 2]
    added = refinement_points(mesh, mass, query, 4)
    assert added.shape == (4, 3)
    assert np.any(np.any((added == bounds[0]) | (added == bounds[1]), axis=1))
    assert np.all(added[:, 2] > 0)


def test_physical_radius_cdf_integrates_partial_cells_at_upper_boundary() -> None:
    points, bounds = box_points()
    mesh = RadiusMesh(points, np.zeros((len(points), 2)), bounds)
    results = []
    for count in (8, 16):
        axes, mass, _ = integrate_radius(mesh, np.zeros((1, 2)), 1, 8, [64, 4, count])
        summary = summarize_radius(axes, mass, 8)
        grid = np.linspace(float(isco_radius(0.98)), 8, 513)
        lower = isco_radius(axes[0])[:, None]
        expected = np.clip((grid - lower) / (8 - lower), 0, 1).mean(axis=0)
        np.testing.assert_allclose(summary["cdf"][0, 0, 2], expected, atol=1e-8)
        results.append(summary["cdf"][0, 0, 2])
    np.testing.assert_allclose(*results, atol=1e-8)


def test_gaussian_spin_posterior_and_refinement_comparison() -> None:
    points, bounds = box_points()
    mesh = RadiusMesh(points, points, bounds)
    axes, mass, _ = integrate_radius(mesh, np.array([[0.49, 0.6, 0.5]]), 0.05, 8, [64, 48, 48])
    summary = summarize_radius(axes, mass, 8)
    assert summary["mean"][0, 0, 0] == pytest.approx(0.49, abs=1e-9)
    assert summary["std"][0, 0, 0] == pytest.approx(0.05, abs=1e-8)
    options = {
        "max_marginal_cdf_change": 0.02,
        "max_mean_shift_in_std": 0.05,
        "max_relative_std_change": 0.05,
    }
    same = compare_radius(summary, summary, options)
    assert np.all(same["passed"])
    displaced = {**summary, "mean": summary["mean"] + 0.02}
    assert not np.any(compare_radius(displaced, summary, options)["passed"])
    coarse_axes, coarse_mass, _ = integrate_radius(
        mesh, np.array([[0.49, 0.6, 0.5]]), 0.05, 8, [32, 24, 24]
    )
    comparison = compare_radius(summarize_radius(coarse_axes, coarse_mass, 8), summary, options)
    assert np.max(np.array(comparison["marginal_cdf_change"])[..., 0]) < 0.01


@pytest.mark.parametrize("confirmation_bias", [0.0, 20.0])
def test_radius_workflow_certification_requires_direct_validation(
    tmp_path: Path, cfg: dict, monkeypatch: pytest.MonkeyPatch, confirmation_bias: float
) -> None:
    project = Path(__file__).resolve().parents[1]
    for name in (
        "src/kerr_sbi/radius.py",
        "src/kerr_sbi/radius_diagnostics.py",
        "scripts/build_radius_reference.py",
        "src/kerr_sbi/reference_mesh.py",
        "src/kerr_sbi/obs_model.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project / name).read_bytes())
    monkeypatch.setattr(
        "kerr_sbi.radius_diagnostics.__file__", str(tmp_path / "src/kerr_sbi/radius_diagnostics.py")
    )
    directory = tmp_path / "runs/pilot"
    data_directory = tmp_path / "data/reference/pilot"
    directory.mkdir(parents=True)
    data_directory.mkdir(parents=True)
    write_json(directory / "experiment.json", {})

    def images(points):
        result = np.zeros((len(points), 64, 64), dtype=np.float32)
        result[:, 0, :3] = points
        return result

    for size in (2, 3):
        points, bounds = box_points(size)
        np.savez(
            data_directory / f"grid_{size}.npz", points=points, images=images(points), bounds=bounds
        )
    source_data = tmp_path / "data/reference/source"
    source_data.mkdir(parents=True)
    source_points = np.array(
        [[0.0, 0.2], [0.98, 0.2], [0.0, 1.0], [0.98, 1.0], [0.49, 0.6], [0.7, 0.8]]
    )
    source_images = images(np.column_stack([source_points, np.zeros(6)]))
    np.savez(
        source_data / "mesh.npz", points=source_points, images=source_images, bounds=bounds[:, :2]
    )
    source_run = tmp_path / "runs/source"
    source_run.mkdir()
    write_json(source_run / "summary.json", {"history": [{"nodes": 4}, {"nodes": 5}, {"nodes": 6}]})
    truth = np.array([[0.5, np.cos(np.deg2rad(45)), t] for t in (0, 1)])
    cases = [
        {
            "truth_index": n,
            "spin": 0.5,
            "inclination_deg": 45.0,
            "radius": float(r),
            "fraction": float(truth[n, 2]),
            "noise_seed": 1,
        }
        for n, r in enumerate(physical_radius(truth, 8))
    ]
    np.savez(
        data_directory / "observations.npz",
        truth=truth,
        images=images(truth),
        observations=images(truth),
    )
    write_json(directory / "cases.json", {"cases": cases})
    calls = []

    def image(a, inclination, radius):
        calls.append((a, inclination, radius))
        point = np.array(
            [[a, np.cos(np.deg2rad(inclination)), (radius - isco_radius(a)) / (8 - isco_radius(a))]]
        )
        result = images(point)[0]
        if len(calls) > 4:
            result[0, 0] += confirmation_bias
        return result

    cache = SimpleNamespace(cfg=cfg, image=image)
    options = {
        "source_run": "source",
        "radius_max": 8.0,
        "pairs": [[0.5, 45.0]],
        "grid_sizes": [2, 3],
        "refinement_counts": [4, 8],
        "quadrature_shapes": [[24, 24, 24], [48, 48, 48]],
        "required_successive_passes": 2,
        "confirmation_seed": 1,
        "confirmation_per_case": 2,
        "boundary_fractions": [0.01, 0.99],
        "max_marginal_cdf_change": 0.02,
        "max_mean_shift_in_std": 0.05,
        "max_relative_std_change": 0.05,
        "max_interpolation_noise_norm": 0.25,
        "figure_dpi": 30,
        "noise_seeds": [1],
    }
    result = analyze_radius(tmp_path, directory, data_directory, cfg, options, cache)
    assert result["all_certified"] == (confirmation_bias == 0)
    assert result["validation"]["boundary_pass"] == (confirmation_bias == 0)
    assert np.all(result["isco_reference"]["certified"]) == (confirmation_bias == 0)
    assert (directory / "spin_radius_joint.png").is_file()
    with np.load(directory / "posteriors.npz") as archive:
        np.testing.assert_allclose(archive["spin_mass"].sum(axis=2), 1)
        np.testing.assert_allclose(
            archive["posterior_mass"].sum(axis=(3, 4)), archive["spin_mass"], atol=1e-8
        )
