from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from kerr_sbi.config import Config, project_path
from kerr_sbi.dataset import file_sha256, load_preprocessed, read_parameters
from kerr_sbi.dummy import build_dummy_dataset, dummy_config, dummy_images
from kerr_sbi.prior import sample_prior


def test_dummy_dataset_is_loadable_reproducible_and_isolated(cfg: Config, tmp_path: Path) -> None:
    cfg["project_root"] = tmp_path
    cfg["dummy_dataset"].update(train_count=8, test_count=3)
    original = deepcopy(cfg)
    production = project_path(cfg, "data") / "train"
    production.mkdir(parents=True)
    sentinel = production / "params.csv"
    sentinel.write_text("existing production data")
    summary = build_dummy_dataset(cfg)
    assert summary["nullgeo_renders"] == 0 and cfg == original
    for split, n in (("train", 8), ("test", 3)):
        isolated = dummy_config(cfg)
        arrays, metadata = load_preprocessed(isolated, split)
        directory = project_path(isolated, "data") / split
        parameters = read_parameters(directory / "params.csv")
        expected = sample_prior(n, cfg["dummy_dataset"][f"{split}_seed"], cfg)
        np.testing.assert_array_equal(parameters, expected)
        np.testing.assert_array_equal(arrays["theta"], expected[:, :2])
        np.testing.assert_array_equal(arrays["idx"], np.arange(n))
        assert arrays["x"].shape == (n, 64, 64) and arrays["x"].dtype == np.float32
        assert np.isfinite(arrays["x"]).all() and (arrays["x"] >= 0).all()
        np.testing.assert_allclose(arrays["x"].sum(axis=(1, 2)), 4096, rtol=1e-7)
        assert metadata["dataset_kind"] == "dummy" and metadata["scientific_use"] is False
        assert metadata["nullgeo"] is None and metadata["n_rendered"] == 0
        assert metadata["params_sha256"] == file_sha256(directory / "params.csv")
    other = deepcopy(cfg)
    other["project_root"] = tmp_path / "repeat"
    build_dummy_dataset(other)
    for split in ("train", "test"):
        left, _ = load_preprocessed(dummy_config(cfg), split)
        right, _ = load_preprocessed(dummy_config(other), split)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])
    with pytest.raises(FileExistsError, match="already exists"):
        build_dummy_dataset(cfg)
    assert sentinel.read_text() == "existing production data"


def test_dummy_image_position_encodes_each_parameter_independently(cfg: Config) -> None:
    parameters = np.array([[0.2, 45.0, 0.4], [0.8, 45.0, 0.4], [0.2, 20.0, 0.9]])
    images = dummy_images(parameters, cfg)
    yy, xx = np.mgrid[:64, :64]
    center_x = (images * xx).sum(axis=(1, 2)) / images.sum(axis=(1, 2))
    center_y = (images * yy).sum(axis=(1, 2)) / images.sum(axis=(1, 2))
    assert center_x[1] > center_x[0] + 20
    assert center_y[2] > center_y[0] + 20
    np.testing.assert_allclose(center_x[0], center_x[2], atol=1e-5)
    np.testing.assert_allclose(center_y[0], center_y[1], atol=1e-5)


def test_dummy_builder_rejects_overlapping_seeds(cfg: Config, tmp_path: Path) -> None:
    cfg["project_root"] = tmp_path
    cfg["dummy_dataset"]["test_seed"] = cfg["dummy_dataset"]["train_seed"]
    with pytest.raises(ValueError, match="different seeds"):
        build_dummy_dataset(cfg)
    assert not project_path(dummy_config(cfg), "data").exists()
