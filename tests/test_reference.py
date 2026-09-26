from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from kerr_sbi.data import Normalization
from kerr_sbi.disk_edge import disk_edge_check
from kerr_sbi.model import Posterior
from kerr_sbi.reference import (
    RenderCache,
    compare_mass,
    compare_samples,
    interpolate_images,
    mass_summary,
    prior_axes,
    prior_particle_reference,
    reference_mass,
    relative_log_likelihood,
)
from kerr_sbi.reference_diagnostics import analyze


def test_interpolation_exact_for_bilinear_images_and_boundaries() -> None:
    a, c = np.linspace(0, 1, 4), np.linspace(0, 1, 5)
    aa, cc = np.meshgrid(a, c, indexing="ij")
    images = np.stack([aa + 2 * cc, aa * cc], axis=-1)
    points = np.array([[0, 0], [1, 1], [0.37, 0.81]])
    result = interpolate_images((a, c), images, points)
    np.testing.assert_allclose(result[:, 0], points[:, 0] + 2 * points[:, 1])
    np.testing.assert_allclose(result[:, 1], points.prod(axis=1))
    with pytest.raises(ValueError, match="outside"):
        interpolate_images((a, c), images, np.array([[1.01, 0.5]]))


def test_likelihood_ratios_equal_direct_gaussian_residuals() -> None:
    rng = np.random.default_rng(12)
    y, images = rng.normal(size=(3, 2, 4)), rng.normal(size=(7, 2, 4))
    result = relative_log_likelihood(y, images, 0.3)
    direct = -np.sum((y[:, None] - images[None]) ** 2, axis=(2, 3)) / (2 * 0.3**2)
    np.testing.assert_allclose(result - result[:, :1], direct - direct[:, :1], atol=1e-12)


def test_reference_recovers_gaussian_and_correct_cosine_prior(cfg: dict) -> None:
    axes = (np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    images = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    a, c, mass = reference_mass(axes, images, np.array([[0.5, 0.5]]), 0.05, 128)
    summary = mass_summary(a, c, mass)
    assert summary["mean"][0][0] == pytest.approx(0.5, abs=1e-10)
    assert summary["std"][0][0] == pytest.approx(0.05, abs=1e-10)
    np.testing.assert_allclose(mass.sum(), 1)
    axes = prior_axes(cfg, 9)
    a, c, mass = reference_mass(axes, np.zeros((9, 9, 2)), np.zeros((1, 2)), 1, 256)
    expected = (cfg["prior"]["a_max"] - cfg["prior"]["a_min"]) / np.sqrt(12)
    assert mass_summary(a, c, mass)["std"][0][0] == pytest.approx(expected, rel=1e-4)
    assert mass_summary(a, c, mass)["mean"][0][1] > 50
    np.testing.assert_allclose(mass, 1 / 256**2)


def test_convergence_detects_displaced_posterior() -> None:
    axis = np.linspace(0.01, 0.99, 10)
    old = np.ones((1, 10, 10)) / 100
    same = compare_mass(axis, axis, old, old)
    assert np.max(same["marginal_cdf_change"]) == 0
    new = old.copy()
    new[:, :5] *= 1.5
    new[:, 5:] *= 0.5
    result = compare_mass(axis, axis, old, new)
    assert result["marginal_cdf_change"][0][0] == pytest.approx(0.25)
    assert result["mean_shift_in_std"][0][0] > 0.4


def test_direct_particle_reference_recovers_uninformative_prior() -> None:
    theta = np.array([[0.1, 10], [0.3, 30], [0.5, 50], [0.7, 70]])
    result = prior_particle_reference(np.zeros((2, 3)), np.zeros((4, 3)), theta, 1)
    np.testing.assert_allclose(result["ess"], 4)
    np.testing.assert_allclose(result["mean"], np.broadcast_to(theta.mean(0), (2, 2)))
    np.testing.assert_allclose(result["std"], np.broadcast_to(theta.std(0), (2, 2)))


def test_sample_comparison_handles_decreasing_inclination_transform() -> None:
    a, c = np.linspace(0.05, 0.95, 10), np.linspace(0.05, 0.95, 10)
    aa, cc = np.meshgrid(a, c, indexing="ij")
    samples = np.stack([aa.ravel(), np.rad2deg(np.arccos(cc.ravel()))], axis=-1)[:, None]
    result = compare_samples(a, c, np.ones((1, 10, 10)) / 100, samples)
    np.testing.assert_allclose(result["marginal_cdf_change"], 0, atol=1e-12)
    np.testing.assert_allclose(result["mean_shift_in_std"], 0, atol=1e-12)
    np.testing.assert_allclose(result["relative_std_change"], 0, atol=1e-12)


def test_cache_reuses_outputs_and_rejects_corruption_and_changed_physics(
    tmp_path: Path, cfg: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_render(a: float, inclination: float, stem: Path, cfg: dict) -> Path:
        calls.append((a, inclination))
        path = stem.with_suffix(".pfm")
        path.write_bytes(b"render")
        return path

    monkeypatch.setattr("kerr_sbi.reference.render", fake_render)
    monkeypatch.setattr("kerr_sbi.reference.simulator_provenance", lambda cfg: {"version": "test"})
    monkeypatch.setattr("kerr_sbi.reference.preprocess_pfm", lambda path, cfg: np.ones((64, 64)))
    cache = RenderCache(tmp_path, cfg)
    cache.image(0.5, 45)
    cache.image(0.5, 45)
    assert len(calls) == 1
    changed = deepcopy(cfg)
    changed["scene"]["r_in"] = 8
    with pytest.raises(ValueError, match="identity"):
        RenderCache(tmp_path, changed)
    next(tmp_path.glob("*.pfm")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        cache.image(0.5, 45)


def test_disk_edge_comparison_recovers_known_information_loss(
    tmp_path: Path, cfg: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LinearCache:
        def __init__(self, directory: Path, config: dict) -> None:
            self.factor = 0.1 if config["scene"]["r_in"] == 8 else 1.0

        def image(self, a: float, inclination: float) -> np.ndarray:
            image = np.zeros((64, 64))
            image[0, 0], image[0, 1] = self.factor * a, inclination
            return image

    monkeypatch.setattr("kerr_sbi.disk_edge.RenderCache", LinearCache)
    settings = {
        "reference": {"figure_dpi": 30},
        "fixed_radius": {
            "r_in": 8,
            "spins": [0.0, 0.49, 0.98],
            "inclinations_deg": [18.0, 68.0],
            "derivative_points": [[0.2, 18.0], [0.8, 68.0]],
            "delta_a": 0.02,
            "delta_incl_deg": 1.0,
        },
    }
    report = disk_edge_check(tmp_path, cfg, settings)
    np.testing.assert_allclose(
        np.array(report["conditional_information_ratio_fixed_over_isco"])[..., 0], 0.01
    )
    np.testing.assert_allclose(np.array(report["joint_std_ratio_fixed_over_isco"])[..., 0], 10)
    assert report["numerically_stable"]
    assert cfg["scene"]["r_in"] == 0


def test_analysis_end_to_end_keeps_linear_noise_and_parameter_jacobian(
    tmp_path: Path, small_model_cfg: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = small_model_cfg
    directory = tmp_path / "reference"
    directory.mkdir()
    grid_directory = tmp_path / "shared_grids"
    grid_directory.mkdir()
    checkpoint = tmp_path / "runs" / "pilot"
    checkpoint.mkdir(parents=True)
    (checkpoint / "latest.json").write_text("{}")
    axes = prior_axes(cfg, 3)
    stats = Normalization(
        (0, 0), (1.7, 1.7), (axes[0][0], axes[1][0]), (axes[0][-1], axes[1][-1]), 1e-6, 1.0
    )
    model = Posterior(cfg, jax.random.PRNGKey(1))
    posterior = SimpleNamespace(
        model=model, normalization=stats, identity={"config": cfg, "effective_sigma_n": 20.0}
    )
    monkeypatch.setattr("kerr_sbi.reference_diagnostics.load_posterior", lambda _: posterior)
    truth = np.array([[0.1, 15], [0.2, 30], [0.4, 45], [0.6, 55], [0.8, 65], [0.9, 75]])
    data_images = np.zeros((6, 64, 64), dtype=np.float32)
    data_images[:, 0, 0] = truth[:, 0]
    data_images[:, 0, 1] = np.cos(np.deg2rad(truth[:, 1]))
    data = SimpleNamespace(theta=truth, x=data_images, idx=np.arange(6), metadata={"arrays": {}})
    monkeypatch.setattr("kerr_sbi.reference_diagnostics.load_split", lambda *_: data)
    for size in (3, 5):
        a, c = prior_axes(cfg, size)
        images = np.zeros((size, size, 64, 64))
        images[:, :, 0, 0] = a[:, None]
        images[:, :, 0, 1] = c[None, :]
        np.savez(grid_directory / f"grid_{size}.npz", a=a, c=c, images=images)
    settings = {
        "reference": {
            "grid_sizes": [3, 5],
            "quadrature_sizes": [8, 16],
            "targets": truth.tolist(),
            "noise_seeds": [104],
            "posterior_samples": 39,
            "posterior_seed": 1,
            "validation_seed": 1,
            "validation_count": 6,
            "max_marginal_cdf_change": 0.02,
            "max_mean_shift_in_std": 0.05,
            "max_relative_std_change": 0.05,
            "max_interpolation_noise_norm": 0.25,
            "figure_dpi": 30,
        }
    }
    report = analyze(directory, checkpoint, settings, grid_directory=grid_directory)
    assert report["grid_directory"] == str(grid_directory)
    assert all(report["grid_pass"])
    assert max(report["interpolation_validation"]["noise_norm_quantiles_50_90_100"]) < 1e-7
    assert np.isfinite(report["npe_quadrature_integral"]).all()
    archive = np.load(directory / "analysis/posteriors.npz")
    assert archive["reference_mass"].shape == (6, 16, 16)
    assert np.any(archive["observations"] < 0)
    np.testing.assert_allclose(archive["npe_mass"].sum(axis=(1, 2)), 1)
