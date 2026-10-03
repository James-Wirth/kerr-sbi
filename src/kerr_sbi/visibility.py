from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import least_squares

UAS_TO_RAD = np.pi / (180 * 3600 * 1e6)


@dataclass(frozen=True)
class VisibilityData:
    time_hours: np.ndarray
    stations: np.ndarray
    uv: np.ndarray
    observed: np.ndarray
    sigma: np.ndarray
    header: str


def load_eht_csv(path: Path, minimum_baseline: float = 0.0) -> VisibilityData:
    if not np.isfinite(minimum_baseline) or minimum_baseline < 0:
        raise ValueError("minimum baseline must be finite and nonnegative")
    with path.open() as stream:
        header, columns = stream.readline().strip(), stream.readline().strip()
    if columns != "#time(UTC),T1,T2,U(lambda),V(lambda),Iamp(Jy),Iphase(d),Isigma(Jy)":
        raise ValueError("unsupported EHT CSV columns")
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", skip_header=2, dtype=None, encoding="utf8")
    )
    uv = np.column_stack((rows["f3"], rows["f4"]))
    sigma, amplitude, phase = rows["f7"], rows["f5"], rows["f6"]
    numeric = np.column_stack((rows["f0"], uv, amplitude, phase, sigma))
    if not np.isfinite(numeric).all() or np.any(sigma <= 0) or np.any(amplitude < 0):
        raise ValueError("invalid EHT measurements or errors")
    mask = np.linalg.norm(uv, axis=1) >= minimum_baseline
    if not mask.any():
        raise ValueError("no baselines survive selection")
    return VisibilityData(
        rows["f0"][mask],
        np.column_stack((rows["f1"], rows["f2"]))[mask],
        uv[mask],
        (amplitude * np.exp(1j * np.deg2rad(phase)))[mask],
        sigma[mask],
        header,
    )


def normalized_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("expected finite two-dimensional image")
    if np.any(image < 0) or image.sum() <= 0:
        raise ValueError("image requires nonnegative intensity and positive flux")
    return image / image.sum()


def image_frequencies(uv: np.ndarray, pixel_uas: float, angle_deg: float) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("expected finite (u,v) coordinates in wavelengths")
    if not np.isfinite(pixel_uas) or pixel_uas <= 0 or not np.isfinite(angle_deg):
        raise ValueError("positive pixel scale and finite orientation required")
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return uv @ rotation * pixel_uas * UAS_TO_RAD * np.array([1, -1])


def translation_phase(uv: np.ndarray, offset_uas: tuple[float, float]) -> np.ndarray:
    offset = np.asarray(offset_uas, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("expected two finite sky offsets")
    return np.exp(-2j * np.pi * (uv @ offset) * UAS_TO_RAD)


def direct_visibility(
    image: np.ndarray,
    uv: np.ndarray,
    pixel_uas: float,
    angle_deg: float = 0.0,
    offset_uas: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    image = normalized_image(image)
    frequencies = image_frequencies(uv, pixel_uas, angle_deg)
    y, x = np.indices(image.shape, dtype=float)
    xy = np.column_stack(
        ((x - (image.shape[1] - 1) / 2).ravel(), (y - (image.shape[0] - 1) / 2).ravel())
    )
    nonzero = image.ravel() > 0
    xy, intensity = xy[nonzero], image.ravel()[nonzero]
    result = np.empty(len(uv), dtype=complex)
    for start in range(0, len(uv), 128):
        stop = start + 128
        result[start:stop] = np.exp(-2j * np.pi * (frequencies[start:stop] @ xy.T)) @ intensity
    return result * np.sinc(frequencies).prod(axis=1) * translation_phase(uv, offset_uas)


class FourierImage:
    def __init__(self, image: np.ndarray, padding: int = 16) -> None:
        self.image = normalized_image(image)
        if not isinstance(padding, int) or padding < 4:
            raise ValueError("Fourier padding must be an integer >= 4")
        self.size = padding * max(image.shape)
        if self.size % 2:
            self.size += 1
        padded = np.zeros((self.size, self.size))
        h, w = image.shape
        y, x = self.size // 2 - h // 2, self.size // 2 - w // 2
        padded[y : y + h, x : x + w] = self.image
        transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
        self.real = spline_filter(transformed.real, order=3)
        self.imag = spline_filter(transformed.imag, order=3)
        self.center_offset = np.array([w // 2 - (w - 1) / 2, h // 2 - (h - 1) / 2])

    def evaluate(
        self,
        uv: np.ndarray,
        pixel_uas: float,
        angle_deg: float = 0.0,
        offset_uas: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        frequencies = image_frequencies(uv, pixel_uas, angle_deg)
        if np.any(np.abs(frequencies) >= 0.45):
            raise ValueError("requested baselines approach or exceed image Nyquist limit")
        indices = (frequencies[:, ::-1] * self.size + self.size / 2).T
        value = map_coordinates(self.real, indices, order=3, prefilter=False)
        value = value + 1j * map_coordinates(self.imag, indices, order=3, prefilter=False)
        center = np.exp(-2j * np.pi * (frequencies @ self.center_offset))
        return (
            value * center * np.sinc(frequencies).prod(axis=1) * translation_phase(uv, offset_uas)
        )


def visibility_distance(first: np.ndarray, second: np.ndarray, sigma: np.ndarray) -> float:
    first, second, sigma = np.asarray(first), np.asarray(second), np.asarray(sigma)
    if first.shape != second.shape or first.shape != sigma.shape or sigma.ndim != 1:
        raise ValueError("matching one-dimensional visibilities and errors required")
    if not all(np.isfinite(value).all() for value in (first, second, sigma)) or np.any(sigma <= 0):
        raise ValueError("finite visibilities and positive errors required")
    return float(np.linalg.norm((first - second) / sigma))


def matched_flux(
    unit_model: np.ndarray, target: np.ndarray, sigma: np.ndarray, bounds: tuple[float, float]
) -> float:
    visibility_distance(unit_model, target, sigma)
    if not 0 < bounds[0] < bounds[1] or not np.isfinite(bounds).all():
        raise ValueError("positive increasing flux bounds required")
    model, data = unit_model / sigma, target / sigma
    denominator = np.vdot(model, model).real
    if denominator <= 0:
        raise ValueError("model has no measured flux")
    return float(np.clip(np.vdot(model, data).real / denominator, *bounds))


def fit_visibility_pair(
    model: FourierImage,
    target: np.ndarray,
    uv: np.ndarray,
    sigma: np.ndarray,
    pixel_uas: float,
    bounds: np.ndarray,
    flux_bounds: tuple[float, float],
    starts: list[list[float]],
    max_evaluations: int,
) -> dict:
    def prediction(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        scale, angle, dx, dy = parameters
        unit = model.evaluate(uv, pixel_uas * scale, angle, (dx, dy))
        flux = matched_flux(unit, target, sigma, flux_bounds)
        return flux * unit, flux

    def residual(parameters: np.ndarray) -> np.ndarray:
        difference = (prediction(parameters)[0] - target) / sigma
        return np.concatenate((difference.real, difference.imag))

    fits = [
        least_squares(
            residual,
            start,
            bounds=bounds,
            x_scale=[1, 30, 10, 10],
            max_nfev=max_evaluations,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: np.linalg.norm(fit.fun))
    value, flux = prediction(best.x)
    return {
        "parameters": best.x.tolist(),
        "flux_jy": flux,
        "distance": visibility_distance(value, target, sigma),
        "optimizer_success": bool(best.success),
        "optimizer_message": best.message,
        "start_distances": [float(np.linalg.norm(fit.fun)) for fit in fits],
        "near_parameter_bound": bool(
            np.any(
                np.minimum(best.x - bounds[0], bounds[1] - best.x) < 1e-4 * (bounds[1] - bounds[0])
            )
        ),
    }


def fit_station_gains(
    target: np.ndarray,
    candidate: np.ndarray,
    data: VisibilityData,
    interval_seconds: float,
    amplitude_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    visibility_distance(target, candidate, data.sigma)
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("positive finite calibration interval required")
    if not 0 < amplitude_bounds[0] < 1 < amplitude_bounds[1]:
        raise ValueError("gain bounds must contain unity")
    groups = np.floor(data.time_hours * 3600 / interval_seconds).astype(int)
    prediction = np.empty_like(candidate)
    records = []
    for group in np.unique(groups):
        selected = groups == group
        names, indices = np.unique(data.stations[selected], return_inverse=True)
        station_indices = indices.reshape(-1, 2)
        n = len(names)
        lower = np.r_[np.full(n, np.log(amplitude_bounds[0])), np.full(n - 1, -np.pi)]
        upper = np.r_[np.full(n, np.log(amplitude_bounds[1])), np.full(n - 1, np.pi)]

        def transformed(
            parameters: np.ndarray,
            n: int = n,
            selected: np.ndarray = selected,
            station_indices: np.ndarray = station_indices,
        ) -> np.ndarray:
            phases = np.r_[0.0, parameters[n:]]
            gains = np.exp(parameters[:n] + 1j * phases)
            return (
                candidate[selected]
                * gains[station_indices[:, 0]]
                * gains[station_indices[:, 1]].conj()
            )

        def residual(
            parameters: np.ndarray, selected: np.ndarray = selected, transformed=transformed
        ) -> np.ndarray:
            delta = (transformed(parameters) - target[selected]) / data.sigma[selected]
            return np.r_[delta.real, delta.imag]

        fit = least_squares(
            residual,
            np.zeros(2 * n - 1),
            bounds=(lower, upper),
            max_nfev=150,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        prediction[selected] = transformed(fit.x)
        records.append(
            {
                "interval": int(group),
                "stations": names.tolist(),
                "amplitudes": np.exp(fit.x[:n]).tolist(),
                "phases_rad": np.r_[0.0, fit.x[n:]].tolist(),
                "success": bool(fit.success),
                "measurements": int(selected.sum()),
            }
        )
    return prediction, {
        "distance": visibility_distance(target, prediction, data.sigma),
        "interval_seconds": interval_seconds,
        "amplitude_bounds": list(amplitude_bounds),
        "all_optimizers_succeeded": all(record["success"] for record in records),
        "max_amplitude_departure": max(
            abs(value - 1) for record in records for value in record["amplitudes"]
        ),
        "groups": records,
    }
