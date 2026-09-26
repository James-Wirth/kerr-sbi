import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.reference import mass_summary, relative_log_likelihood
from kerr_sbi.reference_mesh import ImageMesh


def linear_mesh() -> ImageMesh:
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.3, 0.7]])
    return ImageMesh(points, points, np.array([[0, 0], [1, 1]]))


def test_mesh_exact_linear_images_and_likelihood() -> None:
    mesh = linear_mesh()
    rng = np.random.default_rng(1)
    points = np.concatenate([mesh.points, rng.uniform(size=(25, 2))])
    images = mesh.interpolate(points)
    np.testing.assert_allclose(images, points, atol=1e-14)
    observations = rng.normal(size=(4, 2))
    expected = relative_log_likelihood(observations, images, 0.13)
    np.testing.assert_allclose(
        mesh.log_likelihood(points, observations, 0.13), expected, atol=1e-12
    )
    with pytest.raises(ValueError, match="outside"):
        mesh.interpolate(np.array([[-0.001, 0.5]]))
    with pytest.raises(ValueError, match="finite"):
        mesh.interpolate(np.array([[np.nan, 0.5]]))
    with pytest.raises(ValueError, match="positive"):
        mesh.log_likelihood(points, observations, 0)
    with pytest.raises(ValueError, match="matching"):
        mesh.log_likelihood(points, observations[:, :1], 1)


def test_mesh_likelihood_matches_pixel_residuals_for_nonaffine_images() -> None:
    rng = np.random.default_rng(32)
    bounds = np.array([[0, 0.2], [0.98, 1]])
    unit = np.r_[np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), rng.uniform(size=(20, 2))]
    points = bounds[0] + unit * (bounds[1] - bounds[0])
    images = rng.uniform(0, 100, (len(points), 4, 5))
    mesh = ImageMesh(points, images, bounds)
    queries = bounds[0] + rng.uniform(size=(50, 2)) * (bounds[1] - bounds[0])
    observations = rng.normal(0, 20, (3, 4, 5))
    predicted = mesh.interpolate(queries)
    direct = -np.sum((observations[:, None] - predicted[None]) ** 2, axis=(2, 3)) / (2 * 20**2)
    accelerated = mesh.log_likelihood(queries, observations, 20)
    np.testing.assert_allclose(accelerated - accelerated[:, :1], direct - direct[:, :1], atol=1e-12)


def test_mesh_quadrature_recovers_gaussian_and_uniform_prior() -> None:
    mesh = linear_mesh()
    a, c, mass = mesh.mass(np.array([[0.5, 0.5]]), 0.05, 128)
    summary = mass_summary(a, c, mass)
    assert summary["mean"][0][0] == pytest.approx(0.5, abs=1e-12)
    assert summary["std"][0][0] == pytest.approx(0.05, abs=1e-12)
    mesh = ImageMesh(mesh.points, np.zeros_like(mesh.points), mesh.bounds)
    _, _, mass = mesh.mass(np.zeros((2, 2)), 1, 32)
    np.testing.assert_allclose(mass, 1 / 32**2)
    with pytest.raises(ValueError, match="quadrature"):
        mesh.mass(np.zeros((2, 2)), 1, 1)


def test_mesh_refinement_prioritizes_concentrated_posterior() -> None:
    mesh = linear_mesh()
    mass = np.zeros((1, 32, 32))
    mass[0, 1, 1] = 1
    selected = mesh.refinement_points(mass, 1)
    simplex, _ = mesh.weights(np.array([[1.5 / 32, 1.5 / 32]]))
    np.testing.assert_allclose(
        selected, mesh.points[mesh.triangulation.simplices[simplex]].mean(axis=1)
    )
    assert not np.any(np.all(selected[:, None] == mesh.points, axis=-1))


def test_mesh_rejects_duplicate_nodes_and_bad_bounds() -> None:
    mesh = linear_mesh()
    with pytest.raises(ValueError, match="unique"):
        ImageMesh(np.repeat(mesh.points, 2, axis=0), np.repeat(mesh.points, 2, axis=0), mesh.bounds)
    with pytest.raises(ValueError, match="increasing"):
        ImageMesh(mesh.points, mesh.points, mesh.bounds[::-1])
    with pytest.raises(ValueError, match="outside"):
        ImageMesh(mesh.points + 1, mesh.points, mesh.bounds)


@pytest.mark.parametrize("confirmation_bias", [0.0, 20.0])
def test_refinement_workflow_preserves_source_and_validates_direct_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmation_bias: float
) -> None:
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts/06_refine_reference.py"
    spec = importlib.util.spec_from_file_location("refine_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/06_refine_reference.py"))
    for name in (
        "scripts/06_refine_reference.py",
        "src/kerr_sbi/reference_mesh.py",
        "src/kerr_sbi/reference.py",
        "src/kerr_sbi/reference_diagnostics.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project / name).read_bytes())
    source = tmp_path / "runs/source"
    (source / "analysis").mkdir(parents=True)
    destination = tmp_path / "runs/refined"
    destination.mkdir()
    rng = np.random.default_rng(5)

    def images(points):
        result = np.zeros((len(points), 64, 64), dtype=np.float32)
        result[:, 0, :2] = points
        return result

    def dataset(n):
        points = rng.uniform(0.01, 0.99, (n, 2))
        return SimpleNamespace(
            theta=np.column_stack([points[:, 0], np.rad2deg(np.arccos(points[:, 1]))]),
            x=images(points),
            idx=np.arange(n),
            metadata={"arrays": {}, "nullgeo": {}},
        )

    train, dev = dataset(20), dataset(6)
    monkeypatch.setattr(module, "load_split", lambda cfg, split: train if split == "train" else dev)
    cfg = {}
    checkpoint = tmp_path / "runs/checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "run.json", {"config": cfg, "effective_sigma_n": 20})
    options = {
        "grid_sizes": [3],
        "targets": dev.theta.tolist(),
        "max_marginal_cdf_change": 0.02,
        "max_mean_shift_in_std": 0.05,
        "max_relative_std_change": 0.05,
        "max_interpolation_noise_norm": 0.25,
        "figure_dpi": 30,
    }
    write_json(
        source / "experiment.json", {"checkpoint": "checkpoint", "settings": {"reference": options}}
    )
    grid_path = tmp_path / "data/reference/source/grid_3.npz"
    grid_path.parent.mkdir(parents=True)
    (grid_path.parent / "renders").mkdir()
    write_json(grid_path.parent / "renders/identity.json", {"nullgeo": {}})
    axis = np.linspace(0, 1, 3)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    np.savez(grid_path, a=axis, c=axis, images=images(points).reshape(3, 3, 64, 64))
    write_json(
        source / "analysis/summary.json",
        {
            "grid_sha256": {"3": file_sha256(grid_path)},
            "cases": [{"dev_idx": n, "truth": t.tolist()} for n, t in enumerate(dev.theta)],
        },
    )
    np.savez(
        source / "analysis/posteriors.npz",
        observations=dev.x,
        samples=np.broadcast_to(dev.theta, (30, 6, 2)),
        reference_mass=np.ones((6, 16, 16)) / 256,
    )
    calls = []

    class FakeCache:
        identity = {"nullgeo": {}}

        def __init__(self, directory, config):
            directory.mkdir(parents=True, exist_ok=True)

        def image(self, a, inclination):
            calls.append((a, inclination))
            result = images(np.array([[a, np.cos(np.deg2rad(inclination))]]))[0]
            if len(calls) > 2:
                result[0, 0] += confirmation_bias
            return result

    monkeypatch.setattr(module, "RenderCache", FakeCache)
    settings = {
        "refinement": {
            "training_counts": [10, 20],
            "quadrature_sizes": [8, 16, 32],
            "new_nodes_per_round": [2, 4],
            "confirmation_per_case": 2,
            "confirmation_seed": 1,
            "required_successive_passes": 2,
        }
    }
    original_hash = file_sha256(source / "analysis/posteriors.npz")
    result = module.refine(tmp_path, "source", "refined", settings)
    assert result["reference_certified"] == (confirmation_bias == 0)
    assert result["consecutive_passes"] == 2
    assert len(calls) == 2 + 12
    assert result["interpolation_validation"]["development_pass_count"] == 6
    assert file_sha256(source / "analysis/posteriors.npz") == original_hash
    assert json.loads((destination / "summary.json").read_text())["reference_certified"] == (
        confirmation_bias == 0
    )
    with np.load(destination / "posteriors.npz") as archive:
        np.testing.assert_allclose(archive["reference_mass"].sum(axis=(1, 2)), 1)
    settings["refinement"]["confirmation_seed"] = 2
    with pytest.raises(ValueError, match="identity"):
        module.refine(tmp_path, "source", "refined", settings)
