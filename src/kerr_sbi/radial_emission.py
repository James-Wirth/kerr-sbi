from copy import copy
from math import comb

import numpy as np

from kerr_sbi.transfer import Transfer, require
from kerr_sbi.transfer_fourier import TransferFourier


def radial_basis(radius: np.ndarray, inner: float, outer: float, degree: int = 3) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    require(isinstance(degree, int) and 0 <= degree <= 8, "invalid radial degree")
    require(np.isfinite([inner, outer]).all() and 0 < inner < outer, "invalid radial edges")
    require(
        np.isfinite(radius).all() and np.all((radius >= inner) & (radius <= outer)),
        "radius outside fixed emitting edges",
    )
    t = np.log(radius / inner) / np.log(outer / inner)
    return np.stack(
        [comb(degree, k) * t**k * (1 - t) ** (degree - k) for k in range(degree + 1)],
        axis=-1,
    )


def radial_intensity(transfer: Transfer, q: float, coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)
    require(
        coefficients.ndim == 1
        and 1 <= len(coefficients) <= 9
        and np.isfinite(coefficients).all()
        and np.all(coefficients >= 0)
        and np.any(coefficients > 0),
        "nonnegative finite radial coefficients required",
    )
    values = transfer.intensity(q, 3.0)
    hit = transfer.samples["radius_valid"].astype(bool)
    basis = radial_basis(
        transfer.samples["radius"][hit],
        transfer.metadata["effective_r_in"],
        transfer.metadata["effective_r_out"],
        len(coefficients) - 1,
    )
    values[hit] *= basis @ coefficients
    return values


class RadialFourier:
    def __init__(self, transfer: Transfer, q: float, degree: int = 3, order: int = 12) -> None:
        require(isinstance(degree, int) and 0 <= degree <= 8, "invalid radial degree")
        self.models = [
            TransferFourier(
                transfer,
                order=order,
                values=radial_intensity(transfer, q, np.eye(degree + 1)[k]),
            )
            for k in range(degree + 1)
        ]
        self.moments = np.stack([model.moments for model in self.models])

    def combine(self, coefficients: np.ndarray) -> TransferFourier:
        coefficients = np.asarray(coefficients, dtype=float)
        require(
            coefficients.shape == (len(self.models),)
            and np.isfinite(coefficients).all()
            and np.all(coefficients >= 0)
            and np.any(coefficients > 0),
            "nonnegative finite radial coefficients required",
        )
        model = copy(self.models[0])
        model.moments = np.einsum("k,kij->ij", coefficients, self.moments)
        model.total_mass = float(model.moments[:, 0].sum())
        return model
