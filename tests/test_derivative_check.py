import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from kerr_sbi.config import Config


@pytest.fixture
def derivative_check() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_derivatives.py"
    spec = importlib.util.spec_from_file_location("check_derivatives", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refined_stencils_and_bounds(
    derivative_check: ModuleType, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg["project_root"] = tmp_path
    cfg["scene"].update(height=2, width=2)
    spin_pattern = np.array([1.0, -1.0, 0.0, 0.0])
    incl_pattern = np.array([0.0, 0.0, 1.0, -1.0])
    spins = np.array([0.2, 0.8])
    inclinations = np.array([20.0, 60.0])
    jacobian = np.empty((2, 2, 4, 2))
    for row, a in enumerate(spins):
        jacobian[row, :, :, 0] = 2 * a * spin_pattern
        jacobian[row, :, :, 1] = incl_pattern
    provenance = {"binary_sha256": "binary"}
    metadata = {
        "config": {key: value for key, value in cfg.items() if key != "project_root"},
        "nullgeo": provenance,
        "template_sha256": "template",
    }
    baseline = tmp_path / "baseline.npz"
    np.savez_compressed(
        baseline,
        spins=spins,
        inclinations_deg=inclinations,
        robustness_indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
        jacobian=jacobian,
        centers=np.empty((2, 2, 2, 2)),
        metadata_json=json.dumps(metadata),
    )
    validated = []

    def fake_render(a: float, inclination: float, stem: Path, config: Config) -> Path:
        path = stem.with_suffix(".pfm")
        path.write_text(json.dumps([a, inclination]))
        return path

    def fake_preprocess(path: Path, config: Config) -> np.ndarray:
        a, inclination = json.loads(path.read_text())
        return (1024 + a**2 * spin_pattern + inclination * incl_pattern).reshape(2, 2)

    monkeypatch.setattr(derivative_check, "render", fake_render)
    monkeypatch.setattr(derivative_check, "preprocess_pfm", fake_preprocess)
    monkeypatch.setattr(derivative_check, "simulator_provenance", lambda config: provenance)
    monkeypatch.setattr(derivative_check, "template_sha256", lambda config: "template")
    monkeypatch.setattr(
        derivative_check, "build_scene", lambda a, i, path, config: validated.append((a, i))
    )
    data = derivative_check.render_checks(cfg, baseline)
    assert len(validated) == len(json.loads(str(data["metadata_json"]))["renders"]) == 32
    np.testing.assert_allclose(data["steps"], [[0.02, 1], [0.01, 0.5], [0.005, 0.25]])
    for level in range(3):
        np.testing.assert_allclose(data["jacobians"][level], jacobian.reshape(4, 4, 2), atol=1e-10)
    report = derivative_check.analyze(data)
    np.testing.assert_allclose(report["relative_derivative_changes"], 0, atol=1e-9)
    np.testing.assert_allclose(report["successive_direction_cosines"], 1)
    np.testing.assert_allclose(report["std_ratios_to_baseline"], 1)
    assert not np.asarray(report["finest_pair_flagged"]).any()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("section", "key"),
    [("scene", "supersample"), ("emission", "g_power"), ("observation", "sigma_psf")],
)
def test_reject_changed_physics(
    derivative_check: ModuleType,
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
) -> None:
    metadata = {
        "config": deepcopy(cfg),
        "nullgeo": {"binary_sha256": "binary"},
        "template_sha256": "template",
    }
    monkeypatch.setattr(derivative_check, "template_sha256", lambda config: "template")
    cfg[section][key] += 1
    with pytest.raises(ValueError, match="differs from the baseline"):
        derivative_check.validate_baseline(cfg, metadata, metadata["nullgeo"])


def test_reject_changed_binary_or_template(
    derivative_check: ModuleType, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = {"nullgeo": {"binary_sha256": "old"}, "template_sha256": "old"}
    with pytest.raises(ValueError, match="binary"):
        derivative_check.validate_baseline(cfg, metadata, {"binary_sha256": "new"})
    monkeypatch.setattr(derivative_check, "template_sha256", lambda config: "new")
    with pytest.raises(ValueError, match="template"):
        derivative_check.validate_baseline(cfg, metadata, {"binary_sha256": "old"})


def test_divergent_derivatives_are_flagged(derivative_check: ModuleType, cfg: Config) -> None:
    jacobians = np.broadcast_to(np.eye(2), (3, 1, 2, 2)).copy()
    jacobians[:, 0, :, 0] *= np.array([1, 2, 4])[:, None]
    data = {
        "parameters": np.array([[0.5, 45]]),
        "steps": np.array([[0.02, 1], [0.01, 0.5], [0.005, 0.25]]),
        "jacobians": jacobians,
        "metadata_json": np.asarray(
            json.dumps({"config": {k: v for k, v in cfg.items() if k != "project_root"}})
        ),
    }
    report = derivative_check.analyze(data)
    np.testing.assert_allclose(
        np.array(report["relative_derivative_changes"])[:, 0], [[1, 0], [1, 0]]
    )
    assert report["finest_pair_flagged"] == [[True, False]]
    np.testing.assert_allclose(np.array(report["std_ratios_to_baseline"])[:, 0, 0], [1, 0.5, 0.25])
