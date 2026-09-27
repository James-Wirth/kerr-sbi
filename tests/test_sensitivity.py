import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from kerr_sbi.config import Config


@pytest.fixture
def sensitivity() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "map_sensitivity.py"
    spec = importlib.util.spec_from_file_location("sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_stencils_and_saved_analysis(
    sensitivity: ModuleType, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg["project_root"] = tmp_path
    cfg["scene"].update(height=2, width=2)
    cfg["sensitivity"].update(
        a_min=0.2,
        a_max=0.8,
        a_count=2,
        incl_min_deg=20,
        incl_max_deg=60,
        incl_count=2,
        robustness_indices=[[0, 0]],
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    spin_pattern = np.array([[1.0, -1.0], [0.0, 0.0]])
    incl_pattern = np.array([[0.0, 0.0], [1.0, -1.0]])
    validated = []

    def fake_render(a: float, inclination: float, stem: Path, config: Config) -> Path:
        path = stem.with_suffix(".pfm")
        path.write_text(json.dumps([a, inclination]))
        return path

    def fake_preprocess(path: Path, config: Config) -> np.ndarray:
        a, inclination = json.loads(path.read_text())
        return 1024 + a**2 * spin_pattern + inclination * incl_pattern

    monkeypatch.setattr(sensitivity, "render", fake_render)
    monkeypatch.setattr(sensitivity, "preprocess_pfm", fake_preprocess)
    monkeypatch.setattr(sensitivity, "simulator_provenance", lambda config: {"version": "test"})
    monkeypatch.setattr(sensitivity, "template_sha256", lambda config: "test")
    monkeypatch.setattr(
        sensitivity, "build_scene", lambda a, i, path, config: validated.append((a, i))
    )
    data = sensitivity.render_grid(cfg, config_path)
    metadata = json.loads(str(data["metadata_json"]))
    assert len(validated) == len(metadata["renders"]) == 22
    assert data["marginal_std"].shape == (4, 2, 2, 2)
    for row, a in enumerate(data["spins"]):
        for column in range(2):
            np.testing.assert_allclose(
                data["jacobian"][row, column, :, 0], 2 * a * spin_pattern.ravel()
            )
            np.testing.assert_allclose(data["jacobian"][row, column, :, 1], incl_pattern.ravel())
    np.testing.assert_allclose(data["centers"].sum(axis=(-1, -2)), 4096)
    np.testing.assert_allclose(data["median_peak"], 1064)
    np.testing.assert_allclose(data["noise_levels"], np.array([0.01, 0.03, 0.1, 0.3]) * 1064)
    assert not data["robustness_flagged"].any()
    assert data["robustness_relative_changes"][0] < 1e-9
    assert not data["singular"].any()
    report = sensitivity.summarize(data)
    assert report["grid_points"] == 4
    assert len(report["noise_levels"]) == 4
    json.dumps(report, allow_nan=False)
    archive = tmp_path / "crb_grid.npz"
    np.savez_compressed(archive, **data)
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["jacobian"], data["jacobian"])
        assert json.loads(str(saved["metadata_json"])) == metadata
