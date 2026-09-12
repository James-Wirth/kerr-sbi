import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kerr_sbi import dataset
from kerr_sbi.config import Config, dataset_config, project_path
from kerr_sbi.diagnostics import posterior_metrics, prior_mean, rank_band


def test_prior_reference_and_interval_metrics(cfg: Config) -> None:
    lo, hi = np.cos(np.deg2rad([80, 5]))
    cosine = lo + (np.arange(100000) + 0.5) / 100000 * (hi - lo)
    np.testing.assert_allclose(
        prior_mean(cfg), [0.49, np.rad2deg(np.arccos(cosine)).mean()], atol=1e-5
    )
    rng = np.random.default_rng(24)
    theta = rng.normal(size=(199, 3000, 2))
    truth = rng.normal(size=(3000, 2))
    metrics = posterior_metrics(theta, truth, np.zeros(2))
    for level, coverage in metrics["coverage"].items():
        np.testing.assert_allclose(coverage, float(level), atol=0.035)
    np.testing.assert_allclose(metrics["prior_rmse"], [1, 1], atol=0.05)
    low, high = rank_band(200, 20)
    assert low < 10 < high
    with pytest.raises(ValueError, match="shape"):
        posterior_metrics(theta[:, :, :1], truth, np.zeros(2))


def test_named_dataset_is_independent_of_run_paths(cfg: Config) -> None:
    selected = dataset_config(cfg, "pilot_v1")
    assert selected["paths"]["data"] == "data/datasets/pilot_v1"
    assert selected["paths"]["runs"] == cfg["paths"]["runs"]
    assert cfg["paths"]["data"] == "data"
    with pytest.raises(ValueError):
        dataset_config(cfg, "../runs")


def test_complete_pilot_reuses_simulations_and_saved_diagnostics(
    small_model_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = small_model_cfg
    cfg["paths"]["scene_template"] = str(project_path(cfg, "scene_template"))
    cfg["project_root"] = tmp_path
    cfg["local_pilot_data"].update(train_count=8, dev_count=4, checkpoint_steps=3)
    cfg["local_pilot"].update(batch_size=2, warmup_steps=1, max_steps=6, max_epochs=4)
    cfg["physical_overfit"].update(
        subset_size=4,
        batch_size=2,
        warmup_steps=1,
        max_steps=4,
        target_check_steps=2,
        target_nll_gain=-100.0,
    )
    cfg["development_diagnostics"].update(posterior_samples=19, batch_size=2)
    rendered = []

    def fake_render(a: float, inclination: float, stem: Path, settings: Config) -> Path:
        yy, xx = np.mgrid[:64, :64]
        luminance = np.exp(-((xx - 12 - 35 * a) ** 2 + (yy - 10 - inclination / 2) ** 2) / 32)
        rgb = np.repeat(luminance[..., None], 3, axis=-1)
        output = stem.with_suffix(".pfm")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"PF\n64 64\n-1.0\n" + rgb[::-1].astype("<f4").tobytes())
        rendered.append(output)
        return output

    monkeypatch.setattr(
        dataset,
        "simulator_provenance",
        lambda _: {"version": "fixture", "source_commit": "fixture", "binary_sha256": "fixture"},
    )
    monkeypatch.setattr(dataset, "render", fake_render)
    path = Path(__file__).resolve().parents[1] / "scripts/run_local_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_config", lambda _: cfg)
    arguments = [str(path), "--dataset", "fixture", "--run", "experiment"]
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    assert len(rendered) == 12
    run = tmp_path / "runs/experiment"
    assert len(list(run.glob("checkpoint_*"))) == 2
    report = json.loads((run / "diagnostics/summary.json").read_text())
    assert report["n_observations"] == 4 and report["posterior_samples"] == 19
    assert report["scientific_calibration"] is False and report["split"] == "dev"
    with np.load(run / "diagnostics/samples.npz") as values:
        assert values["theta"].shape == (19, 4, 2)
        assert np.all(values["shuffled_indices"] != np.arange(4))
        assert np.all(values["ranks"] <= 19)
    preserved = [
        *rendered,
        run / "diagnostics/summary.json",
        run / "latest.json",
        tmp_path / "data/datasets/fixture/train/x.npy",
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved}
    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    module.main()
    assert len(rendered) == 12
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved}
    selected = dataset_config(cfg, "fixture")
    with pytest.raises(ValueError, match="different prior seeds"):
        dataset.generate_dataset(selected, "test", 4, 3)
