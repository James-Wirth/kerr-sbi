from collections.abc import Callable

import numpy as np
import pytest

from kerr_sbi.config import Config, load_config


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def full_psf_convolution(cfg: Config) -> Callable[[np.ndarray], np.ndarray]:
    obs = cfg["observation"]
    radius = obs["kernel_size"] // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx**2 + yy**2) / (2 * obs["sigma_psf"] ** 2))
    kernel /= kernel.sum()

    def convolve(image: np.ndarray) -> np.ndarray:
        padded = np.pad(image.astype(np.float64), 2 * radius, mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
        return np.einsum("ijkl,kl->ij", windows, kernel)

    return convolve
