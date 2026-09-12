import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kerr_sbi.config import Config, project_path
from kerr_sbi.dataset import parameter_csv
from kerr_sbi.obs_model import preprocess
from kerr_sbi.persistence import exclusive_lock, file_sha256, write_json
from kerr_sbi.prior import sample_prior


def dummy_config(cfg: Config) -> Config:
    isolated = deepcopy(cfg)
    for name in ("data", "runs", "logs", "figures", "results"):
        isolated["paths"][name] = str(project_path(cfg, name) / "dummy")
    return isolated


def dummy_images(parameters: np.ndarray, cfg: Config) -> np.ndarray:
    height, width = cfg["scene"]["height"], cfg["scene"]["width"]
    if (height, width) != (64, 64):
        raise ValueError("dummy images require the v1 64 by 64 shape")
    settings = cfg["dummy_dataset"]
    margin, scale = settings["center_margin_fraction"], settings["width_fraction"]
    if not np.isfinite([margin, scale]).all() or not 0 < margin < 0.5 or scale <= 0:
        raise ValueError("invalid dummy Gaussian position or width settings")
    prior = cfg["prior"]
    c_lo, c_hi = np.cos(np.deg2rad([prior["incl_max_deg"], prior["incl_min_deg"]]))
    yy, xx = np.mgrid[:height, :width]
    images = []
    for a, _, cosine in parameters:
        a_unit = (a - prior["a_min"]) / (prior["a_max"] - prior["a_min"])
        c_unit = (cosine - c_lo) / (c_hi - c_lo)
        center_x = (margin + (1 - 2 * margin) * a_unit) * (width - 1)
        center_y = (margin + (1 - 2 * margin) * c_unit) * (height - 1)
        radius_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
        intensity = np.exp(-radius_squared / (2 * (scale * min(height, width)) ** 2))
        rgb = np.repeat(intensity.astype(np.float32)[..., None], 3, axis=-1)
        images.append(preprocess(rgb, cfg))
    return np.stack(images)


def build_dummy_dataset(cfg: Config) -> dict[str, Any]:
    settings = cfg["dummy_dataset"]
    parameters = {}
    for split in ("train", "test"):
        count = settings[f"{split}_count"]
        if type(count) is not int or count < 1:
            raise ValueError("dummy split counts must be positive integers")
        parameters[split] = sample_prior(count, settings[f"{split}_seed"], cfg)
    if settings["train_seed"] == settings["test_seed"]:
        raise ValueError("dummy train and test must use different seeds")
    directory = project_path(dummy_config(cfg), "data")
    with exclusive_lock(directory.parent / ".dummy.lock"):
        if directory.exists():
            raise FileExistsError(f"dummy dataset already exists: {directory}")
        with tempfile.TemporaryDirectory(prefix=".dummy-", dir=directory.parent) as temporary:
            stage = Path(temporary) / "dataset"
            for split, theta in parameters.items():
                destination = stage / split
                destination.mkdir(parents=True)
                arrays = {
                    "x.npy": dummy_images(theta, cfg),
                    "theta.npy": theta[:, :2],
                    "idx.npy": np.arange(len(theta), dtype=np.int64),
                }
                for name, values in arrays.items():
                    np.save(destination / name, values, allow_pickle=False)
                params_path = destination / "params.csv"
                params_path.write_text(parameter_csv(theta))
                metadata = {
                    "schema_version": 1,
                    "dataset_kind": "dummy",
                    "scientific_use": False,
                    "purpose": "M4 software development with synthetic Gaussian spots",
                    "date_utc": datetime.now(UTC).isoformat(),
                    "split": split,
                    "seed": settings[f"{split}_seed"],
                    "generator": {
                        "name": "gaussian_position_v1",
                        "source_sha256": file_sha256(Path(__file__)),
                        "numpy_version": np.__version__,
                        "sampler": "numpy.PCG64; interleaved uniform spin and cosine pairs",
                        "center_margin_fraction": settings["center_margin_fraction"],
                        "width_fraction": settings["width_fraction"],
                        "parameter_mapping": "horizontal position: spin; vertical position: cosine",
                    },
                    "nullgeo": None,
                    "template_sha256": None,
                    "prior": deepcopy(cfg["prior"]),
                    "observation": deepcopy(cfg["observation"]),
                    "n_requested": len(theta),
                    "n_examples": len(theta),
                    "n_rendered": 0,
                    "n_failed": 0,
                    "failed_indices": [],
                    "missing_indices": [],
                    "invalid_indices": {},
                    "theta_columns": ["a", "incl_deg"],
                    "cos_i_location": "params.csv, indexed by idx.npy",
                    "params_sha256": file_sha256(params_path),
                    "measurement_space": (
                        "synthetic RGB, shared luminance/PSF/flux preprocessing; no noise or asinh"
                    ),
                    "arrays": {
                        name: {
                            "sha256": file_sha256(destination / name),
                            "shape": list(values.shape),
                            "dtype": str(values.dtype),
                        }
                        for name, values in arrays.items()
                    },
                }
                write_json(destination / "meta.json", metadata)
            stage.rename(directory)
    return {
        "dataset_kind": "dummy",
        "directory": str(directory),
        "train_examples": len(parameters["train"]),
        "test_examples": len(parameters["test"]),
        "nullgeo_renders": 0,
    }
