from math import factorial

import numpy as np

from kerr_sbi.transfer import Transfer, require
from kerr_sbi.visibility import UAS_TO_RAD, image_frequencies, translation_phase


class TransferFourier:
    def __init__(
        self,
        transfer: Transfer,
        emissivity_index: float | None = None,
        g_power: float | None = None,
        order: int = 12,
        sample_batch: int = 262144,
    ) -> None:
        require(isinstance(order, int) and 0 <= order <= 20, "order must be in [0,20]")
        require(isinstance(sample_batch, int) and sample_batch > 0, "invalid sample batch")
        require(
            np.all(transfer.samples["finished"]), "unfinished rays; visibility is not certified"
        )
        self.width, self.height = transfer.metadata["width"], transfer.metadata["height"]
        self.order = order
        self.powers = [(a, degree - a) for degree in range(order + 1) for a in range(degree + 1)]
        self.moments = np.zeros((self.width * self.height, len(self.powers)))
        values = transfer.intensity(emissivity_index, g_power)
        scale = 2 * np.tan(np.deg2rad(transfer.metadata["resolved_scene"]["camera"]["fov_deg"]) / 2)
        for start in range(0, len(values), sample_batch):
            s = transfer.samples[start : start + sample_batch]
            ids = s["pixel_index"].astype(np.intp)
            mass = values[start : start + sample_batch] * s["weight"]
            x = s["screen_u"] * self.width / scale - (ids % self.width - (self.width - 1) / 2)
            y = -s["screen_v"] * self.width / scale - (ids // self.width - (self.height - 1) / 2)
            require(np.all(np.abs(x) <= 0.5 + 1e-12), "sample outside pixel")
            require(np.all(np.abs(y) <= 0.5 + 1e-12), "sample outside pixel")
            xp = [np.ones(len(s))]
            yp = [np.ones(len(s))]
            for degree in range(1, order + 1):
                xp.append(xp[-1] * x / degree)
                yp.append(yp[-1] * y / degree)
            for column, (a, b) in enumerate(self.powers):
                self.moments[:, column] += np.bincount(
                    ids, weights=mass * xp[a] * yp[b], minlength=len(self.moments)
                )
        self.total_mass = float(self.moments[:, 0].sum())
        require(np.isfinite(self.moments).all(), "nonfinite moments")
        y, x = np.indices((self.height, self.width), dtype=float)
        self.centers = np.column_stack(
            ((x - (self.width - 1) / 2).ravel(), (y - (self.height - 1) / 2).ravel())
        )

    def evaluate(
        self,
        uv: np.ndarray,
        fov_uas: float,
        flux_jy: float | None = 1.0,
        angle_deg: float = 0.0,
        offset_uas: tuple[float, float] = (0.0, 0.0),
        baseline_batch: int = 128,
        max_truncation_error: float = 1e-12,
        pixel_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        require(isinstance(baseline_batch, int) and baseline_batch > 0, "invalid baseline batch")
        require(
            np.isfinite(max_truncation_error) and max_truncation_error > 0,
            "positive finite truncation allowance required",
        )
        frequencies = image_frequencies(uv, fov_uas / self.width, angle_deg)
        phase = translation_phase(np.asarray(uv), offset_uas)
        normalization = (fov_uas * UAS_TO_RAD / self.width) ** 2
        if flux_jy is not None:
            require(
                np.isfinite(flux_jy) and flux_jy > 0 and self.total_mass > 0,
                "positive finite flux and nonzero emission required",
            )
            normalization = flux_jy / self.total_mass
        selected = self.moments[:, 0] != 0
        if pixel_mask is not None:
            pixel_mask = np.asarray(pixel_mask)
            require(pixel_mask.shape == selected.shape and pixel_mask.dtype == bool, "invalid mask")
            selected &= pixel_mask
        mass = self.moments[selected, 0].sum() * normalization
        radius = (0.5 + 1e-12) * 2 * np.pi * np.abs(frequencies).sum(axis=1)
        with np.errstate(over="ignore"):
            bound = mass * radius ** (self.order + 1) / factorial(self.order + 1)
        require(np.isfinite(bound).all(), "nonfinite truncation bound")
        require(np.all(bound <= max_truncation_error), "Fourier truncation allowance exceeded")
        moments = np.ascontiguousarray(self.moments[selected].T)
        centers = self.centers[selected]
        result = np.zeros(len(frequencies), dtype=complex)
        for start in range(0, len(result), baseline_batch):
            f = frequencies[start : start + baseline_batch]
            coefficients = np.column_stack(
                [
                    (-2j * np.pi * f[:, 0]) ** a * (-2j * np.pi * f[:, 1]) ** b
                    for a, b in self.powers
                ]
            )
            polynomial = coefficients.real @ moments + 1j * (coefficients.imag @ moments)
            argument = f[:, :1] * centers[:, 0] + f[:, 1:] * centers[:, 1]
            result[start : start + baseline_batch] = np.sum(
                polynomial * np.exp(-2j * np.pi * argument), axis=1
            )
        return result * normalization * phase, bound
