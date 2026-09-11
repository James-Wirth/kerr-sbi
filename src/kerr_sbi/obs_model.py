import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kerr_sbi.config import Config
from kerr_sbi.pfm import read_pfm

LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected an RGB array with shape (height, width, 3)")
    if not np.all(np.isfinite(rgb)) or np.any(rgb < 0):
        raise ValueError("linear radiance must be finite and nonnegative")
    return rgb @ LUMINANCE_WEIGHTS


def gaussian_blur(image: np.ndarray, sigma_psf: float, kernel_size: int) -> np.ndarray:
    if not math.isfinite(sigma_psf) or sigma_psf <= 0:
        raise ValueError("sigma_psf must be finite and positive")
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 1 or not np.all(np.isfinite(image)):
        raise ValueError("expected a finite, nonempty two-dimensional image")
    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_psf) ** 2)
    kernel /= kernel.sum()
    blurred = image.astype(np.float64)
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        blurred = np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"),
            axis,
            np.pad(blurred, padding, mode="constant"),
        )
    return blurred.astype(np.float32)


def preprocess(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    expected_shape = (cfg["scene"]["height"], cfg["scene"]["width"], 3)
    if rgb.shape != expected_shape:
        raise ValueError(f"expected image shape {expected_shape}, got {rgb.shape}")
    obs = cfg["observation"]
    blurred = gaussian_blur(luminance(rgb), obs["sigma_psf"], obs["kernel_size"])
    total = blurred.sum(dtype=np.float64)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("cannot normalize an image without positive finite flux")
    target = obs["flux_sum"]
    if not math.isfinite(target) or target <= 0:
        raise ValueError("flux_sum must be finite and positive")
    return (blurred.astype(np.float64) * (target / total)).astype(np.float32)


def preprocess_pfm(path: str | Path, cfg: Config) -> np.ndarray:
    return preprocess(read_pfm(path), cfg)


def add_noise(x: jax.Array, key: jax.Array, sigma_n: float) -> jax.Array:
    if not math.isfinite(sigma_n) or sigma_n < 0:
        raise ValueError("sigma_n must be finite and nonnegative")
    x = jnp.asarray(x, dtype=jnp.float32)
    return x + sigma_n * jax.random.normal(key, x.shape, dtype=x.dtype)


def network_transform(x_obs: jax.Array, s: float) -> jax.Array:
    if not math.isfinite(s) or s <= 0:
        raise ValueError("asinh scale s must be finite and positive")
    return jnp.arcsinh(x_obs / s)


def observe(x: jax.Array, key: jax.Array, sigma_n: float, s: float) -> jax.Array:
    return network_transform(add_noise(x, key, sigma_n), s)
