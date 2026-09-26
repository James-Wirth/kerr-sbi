import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from kerr_sbi.obs_model import preprocess_pfm
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.reference import RenderCache
from kerr_sbi.simulator import render

PRIOR_NAMES = ("uniform_radius", "log_uniform_radius", "flat_joint")


def isco_radius(spin: np.ndarray | float) -> np.ndarray:
    spin = np.asarray(spin, dtype=np.float64)
    if not np.isfinite(spin).all() or np.any(spin < 0) or np.any(spin > 0.98):
        raise ValueError("spin must be finite and in [0, 0.98]")
    z1 = 1 + np.cbrt(1 - spin**2) * (np.cbrt(1 + spin) + np.cbrt(1 - spin))
    z2 = np.sqrt(3 * spin**2 + z1**2)
    return 3 + z2 - np.sqrt(np.maximum((3 - z1) * (3 + z1 + 2 * z2), 0))


def physical_radius(points: np.ndarray, radius_max: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or np.any(points[:, 2] < 0)
        or np.any(points[:, 2] > 1)
        or not np.isfinite(radius_max)
        or radius_max <= 6
    ):
        raise ValueError("expected finite (a, cos_i, radius_fraction) points and radius_max > 6")
    inner = isco_radius(points[:, 0])
    return inner + points[:, 2] * (radius_max - inner)


def radius_prior_weights(points: np.ndarray, radius_max: float) -> np.ndarray:
    radius = physical_radius(points, radius_max)
    inner = isco_radius(points[:, 0])
    width = radius_max - inner
    return np.stack([np.ones(len(points)), width / (radius * np.log(radius_max / inner)), width])


class RadiusCache(RenderCache):
    def __init__(self, directory: Path, cfg: dict) -> None:
        if cfg["scene"]["mass"] != 1 or cfg["scene"]["r_in"] != 0:
            raise ValueError("radius cache requires mass=1 and the baseline ISCO configuration")
        super().__init__(directory, cfg)

    def image(self, a: float, inclination: float, radius: float) -> np.ndarray:
        if (
            not np.isfinite(radius)
            or radius < float(isco_radius(a)) - 1e-12
            or radius >= self.cfg["scene"]["r_out"]
        ):
            raise ValueError("physical radius must be at least ISCO and below the outer edge")
        parameters = [float(f"{value:.15g}") for value in (a, inclination, radius)]
        key = hashlib.sha256(json.dumps(parameters).encode()).hexdigest()[:24]
        stem = self.directory / key
        record, pfm = stem.with_suffix(".json"), stem.with_suffix(".pfm")
        if record.exists():
            metadata = json.loads(record.read_text())
            if metadata["parameters"] != parameters or file_sha256(pfm) != metadata["sha256"]:
                raise ValueError("cached radius render is corrupt")
            return preprocess_pfm(pfm, self.cfg)
        cfg = deepcopy(self.cfg)
        cfg["scene"]["r_in"] = parameters[2]
        started = time.perf_counter()
        path = render(parameters[0], parameters[1], stem, cfg)
        image = preprocess_pfm(path, cfg)
        write_json(
            record,
            {
                "parameters": parameters,
                "sha256": file_sha256(path),
                "seconds": time.perf_counter() - started,
            },
        )
        return image


class RadiusMesh:
    def __init__(self, points: np.ndarray, images: np.ndarray, bounds: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        images = np.asarray(images, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 3
            or self.bounds.shape != (2, 3)
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[1] <= self.bounds[0])
            or not np.isfinite(self.points).all()
            or images.ndim < 2
            or len(images) != len(points)
            or not np.isfinite(images).all()
            or np.any(self.points < self.bounds[0])
            or np.any(self.points > self.bounds[1])
            or len(np.unique(self.points, axis=0)) != len(points)
        ):
            raise ValueError(
                "mesh requires unique finite 3D points and matching images within bounds"
            )
        self.width = self.bounds[1] - self.bounds[0]
        self.image_shape = images.shape[1:]
        self.images = images.reshape(len(images), -1)
        self.triangulation = Delaunay((self.points - self.bounds[0]) / self.width)
        self.gram = np.empty((len(self.triangulation.simplices), 4, 4))
        for start in range(0, len(self.gram), 128):
            vertices = self.images[self.triangulation.simplices[start : start + 128]]
            self.gram[start : start + 128] = np.einsum("svk,swk->svw", vertices, vertices)

    def weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("expected finite 3D query points")
        unit = (points - self.bounds[0]) / self.width
        simplex = self.triangulation.find_simplex(unit)
        if np.any(simplex < 0):
            raise ValueError("interpolation outside radius mesh")
        transform = self.triangulation.transform[simplex]
        weights = np.einsum("nij,nj->ni", transform[:, :3], unit - transform[:, 3])
        if not np.isfinite(weights).all():
            raise ValueError("singular interpolation simplex at query point")
        return simplex, np.column_stack([weights, 1 - weights.sum(axis=1)])

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        simplex, weights = self.weights(points)
        result = np.empty((len(points), self.images.shape[1]))
        for start in range(0, len(points), 128):
            stop = start + 128
            result[start:stop] = np.einsum(
                "nv,nvk->nk",
                weights[start:stop],
                self.images[self.triangulation.simplices[simplex[start:stop]]],
            )
        return result.reshape((len(points), *self.image_shape))

    def log_likelihood(
        self, points: np.ndarray, observations: np.ndarray, sigma: float
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64)
        if (
            not np.isfinite(sigma)
            or sigma <= 0
            or observations.shape[1:] != self.image_shape
            or not np.isfinite(observations).all()
        ):
            raise ValueError("finite matching observations and positive noise required")
        dot = observations.reshape(len(observations), -1) @ self.images.T
        result = np.empty((len(observations), len(points)))
        for start in range(0, len(points), 16384):
            stop = start + 16384
            simplex, weights = self.weights(points[start:stop])
            linear = np.einsum("onv,nv->on", dot[:, self.triangulation.simplices[simplex]], weights)
            norm = np.einsum("nv,nvw,nw->n", weights, self.gram[simplex], weights)
            result[:, start:stop] = (linear - 0.5 * norm) / sigma**2
        return result


def quadrature_points(bounds: np.ndarray, shape: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
    if len(shape) != 3 or any(n < 2 for n in shape):
        raise ValueError("three quadrature sizes >= 2 required")
    axes = [
        bounds[0, j] + (np.arange(n) + 0.5) / n * (bounds[1, j] - bounds[0, j])
        for j, n in enumerate(shape)
    ]
    return axes, np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def integrate_radius(
    mesh: RadiusMesh, observations: np.ndarray, sigma: float, radius_max: float, shape: list[int]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    axes, points = quadrature_points(mesh.bounds, shape)
    likelihood = mesh.log_likelihood(points, observations, sigma)
    likelihood -= likelihood.max(axis=1, keepdims=True)
    base = np.exp(likelihood)
    priors = radius_prior_weights(points, radius_max)
    mass = np.empty((len(PRIOR_NAMES), len(observations), *shape), dtype=np.float32)
    for j, weights in enumerate(priors):
        weighted = base * weights
        weighted /= weighted.sum(axis=1, keepdims=True)
        mass[j] = weighted.reshape(len(observations), *shape)
    return axes, mass, points
