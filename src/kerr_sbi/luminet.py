import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.spatial import cKDTree


def impact_coordinates(u: np.ndarray, v: np.ndarray, distance: float) -> np.ndarray:
    if distance <= 2:
        raise ValueError("static observer must be outside 2M")
    scale = distance / np.sqrt((1 - 2 / distance) * (1 + u * u + v * v))
    return np.column_stack((u * scale, v * scale))


def first_crossing_angle(beta: np.ndarray, inclination: float) -> np.ndarray:
    angle = np.deg2rad(inclination)
    if not 0 < inclination < 90:
        raise ValueError("inclination must be strictly between 0 and 90 degrees")
    return np.arctan2(np.full_like(beta, np.cos(angle)), -np.sin(beta) * np.sin(angle))


def orbit_radius(impact: float, angle: float, distance: float, tolerance: float) -> float:
    initial = 0.0 if np.isinf(distance) else 1 / distance
    slope_squared = 1 / impact**2 - initial**2 + 2 * initial**3
    if slope_squared <= 0:
        raise ValueError("ray is not initially inward")

    def derivative(t: float, y: np.ndarray) -> tuple:
        return y[1], -y[0] + 3 * y[0] ** 2

    def capture(t: float, y: np.ndarray) -> float:
        return y[0] - 0.5

    def escape(t: float, y: np.ndarray) -> float:
        return y[0]

    capture.terminal = True
    capture.direction = 1
    escape.terminal = True
    escape.direction = -1
    solution = solve_ivp(
        derivative,
        (0, angle),
        [initial, np.sqrt(slope_squared)],
        method="DOP853",
        rtol=tolerance,
        atol=tolerance,
        events=(capture, escape),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    if len(solution.t_events[0]):
        return 2.0
    if len(solution.t_events[1]):
        return np.inf
    return 1 / solution.y[0, -1]


def ring_reference(
    radii: np.ndarray, beta: np.ndarray, inclination: float, distance: float, tolerance: float
) -> np.ndarray:
    angles = first_crossing_angle(beta, inclination)
    result = np.empty((len(radii), len(beta)))
    for j, radius in enumerate(radii):
        for k, angle in enumerate(angles):
            result[j, k] = brentq(
                lambda b, angle=angle, radius=radius: (
                    orbit_radius(b, angle, distance, tolerance) - radius
                ),
                0.5,
                2 * radius,
                xtol=tolerance * 10,
            )
    return result


def redshift(radius: np.ndarray, x: np.ndarray, inclination: float) -> np.ndarray:
    radius = np.asarray(radius)
    if np.any(radius <= 3):
        raise ValueError("timelike circular orbit requires r>3M")
    return (1 + x * np.sin(np.deg2rad(inclination)) / radius**1.5) / np.sqrt(1 - 3 / radius) - 1


def corrected_redshift(g: np.ndarray, distance: float) -> np.ndarray:
    return 1 / (g * np.sqrt(1 - 2 / distance)) - 1


def reconstruct_rings(
    samples: np.ndarray,
    distance: float,
    beta: np.ndarray,
    radii: np.ndarray,
    reference: np.ndarray,
    neighbours: int,
    max_distance: float,
    max_span: float,
    max_offset: float,
) -> dict:
    xy = impact_coordinates(samples["screen_u"], samples["screen_v"], distance)
    direction = np.column_stack((np.cos(beta), np.sin(beta)))
    points = (reference[:, :, None] * direction[None, :, :]).reshape(-1, 2)
    distances, indices = cKDTree(xy).query(points, k=neighbours)
    chosen = samples[indices]
    local = xy[indices] - points[:, None, :]
    matrix = np.concatenate((np.ones((*local.shape[:2], 1)), local), axis=2)
    valid = np.all(
        chosen["has_intersection"].astype(bool)
        & chosen["radius_valid"].astype(bool)
        & chosen["g_valid"].astype(bool)
        & chosen["g_available"].astype(bool)
        & chosen["finished"].astype(bool),
        axis=1,
    )
    valid &= distances[:, -1] < max_distance
    valid &= np.ptp(chosen["radius"], axis=1) < max_span
    radius_fit = np.full((len(points), 3), np.nan)
    shift_fit = radius_fit.copy()
    for j in np.flatnonzero(valid):
        radius_fit[j] = np.linalg.lstsq(matrix[j], chosen["radius"][j], rcond=None)[0]
        shift_fit[j] = np.linalg.lstsq(
            matrix[j], corrected_redshift(chosen["g"][j], distance), rcond=None
        )[0]
    directions = np.tile(direction, (len(radii), 1))
    gradient = np.sum(radius_fit[:, 1:] * directions, axis=1)
    offset = (np.repeat(radii, len(beta)) - radius_fit[:, 0]) / gradient
    valid &= (gradient > 0) & (np.abs(offset) < max_offset)
    shift = shift_fit[:, 0] + offset * np.sum(shift_fit[:, 1:] * directions, axis=1)
    shape = reference.shape
    return dict(
        b=(reference.ravel() + offset).reshape(shape),
        z=shift.reshape(shape),
        valid=valid.reshape(shape),
        indices=indices,
        distances=distances,
        radius_fit=radius_fit,
        shift_fit=shift_fit,
        offset=offset.reshape(shape),
    )


def residual_metrics(values: np.ndarray) -> dict:
    if not np.all(np.isfinite(values)):
        return {"rms": None, "max": None}
    return {"rms": float(np.sqrt(np.mean(values**2))), "max": float(np.max(np.abs(values)))}
