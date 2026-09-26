import math

import numpy as np
from scipy.spatial import Delaunay


class ImageMesh:
    def __init__(self, points: np.ndarray, images: np.ndarray, bounds: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        images = np.asarray(images, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or self.bounds.shape != (2, 2)
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[1] <= self.bounds[0])
            or not np.isfinite(points).all()
            or images.ndim < 2
            or len(images) != len(points)
            or not np.isfinite(images).all()
        ):
            raise ValueError("mesh requires finite points, matching images and increasing bounds")
        if np.any(points < self.bounds[0]) or np.any(points > self.bounds[1]):
            raise ValueError("mesh nodes outside bounds")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("mesh nodes must be unique")
        self.points, self.image_shape = points, images.shape[1:]
        self.images = images.reshape(len(images), -1)
        self.width = self.bounds[1] - self.bounds[0]
        self.triangulation = Delaunay((points - self.bounds[0]) / self.width)
        vertices = self.images[self.triangulation.simplices]
        self.gram = np.einsum("svk,swk->svw", vertices, vertices)

    def weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("expected finite (a, cos_i) points")
        unit = (points - self.bounds[0]) / self.width
        simplex = self.triangulation.find_simplex(unit)
        if np.any(simplex < 0):
            raise ValueError("interpolation outside mesh")
        transform = self.triangulation.transform[simplex]
        weights = np.einsum("nij,nj->ni", transform[:, :2], unit - transform[:, 2])
        return simplex, np.column_stack([weights, 1 - weights.sum(axis=1)])

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        simplex, weights = self.weights(points)
        images = np.einsum(
            "nv,nvk->nk", weights, self.images[self.triangulation.simplices[simplex]]
        )
        return images.reshape((len(points), *self.image_shape))

    def log_likelihood(
        self, points: np.ndarray, observations: np.ndarray, sigma: float
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("noise must be positive")
        if observations.shape[1:] != self.image_shape or not np.isfinite(observations).all():
            raise ValueError("observations must have matching finite pixels")
        simplex, weights = self.weights(points)
        dot = observations.reshape(len(observations), -1) @ self.images.T
        linear = np.einsum("onv,nv->on", dot[:, self.triangulation.simplices[simplex]], weights)
        norm = np.einsum("nv,nvw,nw->n", weights, self.gram[simplex], weights)
        return (linear - 0.5 * norm) / sigma**2

    def mass(
        self, observations: np.ndarray, sigma: float, size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, c, points = midpoint_points(self.bounds, size)
        likelihood = self.log_likelihood(points, observations, sigma)
        mass = np.exp(likelihood - likelihood.max(axis=1, keepdims=True))
        mass /= mass.sum(axis=1, keepdims=True)
        return a, c, mass.reshape(-1, size, size)

    def refinement_points(self, mass: np.ndarray, count: int) -> np.ndarray:
        if mass.ndim != 3 or mass.shape[1] != mass.shape[2] or count < 1:
            raise ValueError("expected square posterior masses and a positive refinement count")
        if not np.isfinite(mass).all() or np.any(mass < 0):
            raise ValueError("posterior masses must be finite and nonnegative")
        _, _, points = midpoint_points(self.bounds, mass.shape[1])
        simplex, _ = self.weights(points)
        scores = np.stack(
            [np.bincount(simplex, weights=row.ravel(), minlength=len(self.gram)) for row in mass]
        ).max(axis=0)
        selected = np.argsort(-scores, kind="stable")[:count]
        return self.points[self.triangulation.simplices[selected]].mean(axis=1)


def midpoint_points(bounds: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("quadrature needs at least two cells")
    a, c = (bounds[0] + (np.arange(size)[:, None] + 0.5) / size * np.diff(bounds, axis=0)).T
    points = np.stack(np.meshgrid(a, c, indexing="ij"), axis=-1).reshape(-1, 2)
    return a, c, points
