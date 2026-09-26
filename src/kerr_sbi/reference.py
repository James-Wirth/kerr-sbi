import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from kerr_sbi.config import Config
from kerr_sbi.obs_model import preprocess_pfm
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.simulator import render, template_sha256


class RenderCache:
    def __init__(self, directory: Path, cfg: Config) -> None:
        self.directory, self.cfg = directory, cfg
        directory.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "config": {k: cfg[k] for k in ("prior", "scene", "emission", "observation")},
            "nullgeo": simulator_provenance(cfg),
            "template_sha256": template_sha256(cfg),
            "preprocessing_sha256": file_sha256(Path(__file__).with_name("obs_model.py")),
            "pfm_reader_sha256": file_sha256(Path(__file__).with_name("pfm.py")),
        }
        manifest = directory / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.identity:
                raise ValueError("render cache identity differs")
        else:
            if list(directory.iterdir()):
                raise ValueError("render cache has no identity")
            write_json(manifest, self.identity)

    def image(self, a: float, inclination: float) -> np.ndarray:
        parameters = [float(f"{a:.15g}"), float(f"{inclination:.15g}")]
        key = hashlib.sha256(json.dumps(parameters).encode()).hexdigest()[:24]
        stem = self.directory / key
        record = stem.with_suffix(".json")
        pfm = stem.with_suffix(".pfm")
        if record.exists():
            metadata = json.loads(record.read_text())
            if metadata["parameters"] != parameters or file_sha256(pfm) != metadata["sha256"]:
                raise ValueError("cached render is corrupt")
            return preprocess_pfm(pfm, self.cfg)
        started = time.perf_counter()
        path = render(*parameters, stem, self.cfg)
        image = preprocess_pfm(path, self.cfg)
        write_json(
            record,
            {
                "parameters": parameters,
                "sha256": file_sha256(path),
                "seconds": time.perf_counter() - started,
            },
        )
        return image


def prior_axes(cfg: Config, size: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 3:
        raise ValueError("grid needs at least three nodes per axis")
    prior = cfg["prior"]
    return (
        np.linspace(prior["a_min"], prior["a_max"], size),
        np.linspace(
            np.cos(np.deg2rad(prior["incl_max_deg"])),
            np.cos(np.deg2rad(prior["incl_min_deg"])),
            size,
        ),
    )


def interpolate_images(
    axes: tuple[np.ndarray, np.ndarray], images: np.ndarray, points: np.ndarray
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("expected finite (a, cos_i) points")
    if images.shape[:2] != tuple(len(axis) for axis in axes):
        raise ValueError("image grid differs from axes")
    indices, weights = [], []
    for dimension, axis in enumerate(axes):
        if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("axes must increase")
        values = points[:, dimension]
        if np.any(values < axis[0] - 1e-14) or np.any(values > axis[-1] + 1e-14):
            raise ValueError("interpolation outside grid")
        values = np.clip(values, axis[0], axis[-1])
        index = np.clip(np.searchsorted(axis, values, side="right") - 1, 0, len(axis) - 2)
        indices.append(index)
        weights.append((values - axis[index]) / (axis[index + 1] - axis[index]))
    i, j = indices
    shape = (-1,) + (1,) * (images.ndim - 2)
    u, v = (weight.reshape(shape) for weight in weights)
    return (
        (1 - u) * (1 - v) * images[i, j]
        + u * (1 - v) * images[i + 1, j]
        + (1 - u) * v * images[i, j + 1]
        + u * v * images[i + 1, j + 1]
    )


def relative_log_likelihood(
    observations: np.ndarray, images: np.ndarray, sigma: float
) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("noise must be positive")
    observations = np.asarray(observations, dtype=np.float64)
    images = np.asarray(images, dtype=np.float64)
    if observations.shape[1:] != images.shape[1:] or not (
        np.isfinite(observations).all() and np.isfinite(images).all()
    ):
        raise ValueError("likelihood images must have matching finite pixels")
    y, f = observations.reshape(len(observations), -1), images.reshape(len(images), -1)
    return (y @ f.T - 0.5 * np.sum(f * f, axis=1)) / sigma**2


def reference_mass(
    axes: tuple[np.ndarray, np.ndarray],
    images: np.ndarray,
    observations: np.ndarray,
    sigma: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("quadrature needs at least two cells")
    a, c = [axis[0] + (np.arange(size) + 0.5) / size * (axis[-1] - axis[0]) for axis in axes]
    points = np.stack(np.meshgrid(a, c, indexing="ij"), axis=-1).reshape(-1, 2)
    likelihood = np.empty((len(observations), len(points)))
    for start in range(0, len(points), 256):
        stop = start + 256
        likelihood[:, start:stop] = relative_log_likelihood(
            observations, interpolate_images(axes, images, points[start:stop]), sigma
        )
    mass = np.exp(likelihood - likelihood.max(axis=1, keepdims=True))
    mass /= mass.sum(axis=1, keepdims=True)
    return a, c, mass.reshape(-1, size, size)


def mass_summary(a: np.ndarray, c: np.ndarray, mass: np.ndarray) -> dict[str, Any]:
    inclination = np.rad2deg(np.arccos(c))
    marginals = (mass.sum(axis=2), mass.sum(axis=1))
    values = (a, inclination)
    means, stds, intervals = [], [], []
    for marginal, support in zip(marginals, values, strict=True):
        mean = marginal @ support
        means.append(mean)
        stds.append(np.sqrt(np.maximum(marginal @ support**2 - mean**2, 0)))
        order = np.argsort(support)
        cdf = np.cumsum(marginal[:, order], axis=1)
        intervals.append(
            [np.interp([0.05, 0.5, 0.95], row, support[order]).tolist() for row in cdf]
        )
    return {
        "mean": np.stack(means, axis=1).tolist(),
        "std": np.stack(stds, axis=1).tolist(),
        "quantiles_05_50_95": np.stack(intervals, axis=1).tolist(),
    }


def compare_mass(a: np.ndarray, c: np.ndarray, old: np.ndarray, new: np.ndarray) -> dict:
    if old.shape != new.shape:
        raise ValueError("comparison needs matching quadrature")
    first, second = mass_summary(a, c, old), mass_summary(a, c, new)
    std = np.asarray(second["std"])
    return {
        "marginal_cdf_change": np.stack(
            [
                np.max(np.abs(np.cumsum(old.sum(axis=k) - new.sum(axis=k), axis=1)), axis=1)
                for k in (2, 1)
            ],
            axis=1,
        ).tolist(),
        "mean_shift_in_std": (np.abs(np.array(first["mean"]) - second["mean"]) / std).tolist(),
        "relative_std_change": (np.abs(np.array(first["std"]) / std - 1)).tolist(),
    }


def prior_particle_reference(
    observations: np.ndarray, images: np.ndarray, theta: np.ndarray, sigma: float
) -> dict:
    log_weights = relative_log_likelihood(observations, images, sigma)
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    mean = weights @ theta
    residual = theta[None] - mean[:, None]
    variance = np.sum(weights[..., None] * residual**2, axis=1)
    mean_se = np.sqrt(np.sum(weights[..., None] ** 2 * residual**2, axis=1))
    variance_se = np.sqrt(
        np.sum(weights[..., None] ** 2 * (residual**2 - variance[:, None]) ** 2, axis=1)
    )
    return {
        "particles": len(images),
        "ess": (1 / np.sum(weights**2, axis=1)).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "mean_mc_standard_error": mean_se.tolist(),
        "std_mc_standard_error": (variance_se / (2 * np.sqrt(variance))).tolist(),
        "interpretation": (
            "Direct likelihood on independent prior draws; no image interpolation. "
            "Self-normalized importance sampling standard errors are asymptotic estimates. "
            "Low effective sample sizes make this only a rough cross-check."
        ),
    }


def compare_samples(a: np.ndarray, c: np.ndarray, mass: np.ndarray, samples: np.ndarray) -> dict:
    summary = mass_summary(a, c, mass)
    cdf_changes = []
    for case in range(len(mass)):
        changes = []
        for dimension, (axis, marginal) in enumerate(
            ((a, mass[case].sum(axis=1)), (c, mass[case].sum(axis=0)))
        ):
            values = (
                samples[:, case, 0] if dimension == 0 else np.cos(np.deg2rad(samples[:, case, 1]))
            )
            edges = np.r_[axis[0] - (axis[1] - axis[0]) / 2, axis + (axis[1] - axis[0]) / 2]
            counts = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)[0]
            changes.append(float(np.max(np.abs(np.cumsum(counts / counts.sum() - marginal)))))
        cdf_changes.append(changes)
    reference_std = np.asarray(summary["std"])
    return {
        "marginal_cdf_change": cdf_changes,
        "mean_shift_in_std": (
            np.abs(samples.mean(axis=0) - summary["mean"]) / reference_std
        ).tolist(),
        "relative_std_change": (np.abs(samples.std(axis=0) / reference_std - 1)).tolist(),
        "normalization": (
            "Mean shifts and width changes are relative to reference standard deviations; "
            "NPE uses posterior samples to avoid boundary quadrature error."
        ),
    }
