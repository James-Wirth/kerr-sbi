import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerr_sbi.config import Config
from kerr_sbi.obs_model import add_noise, gaussian_blur, luminance, network_transform, observe
from kerr_sbi.obs_model import preprocess as preprocess_image


def test_luminance_primaries() -> None:
    np.testing.assert_allclose(
        luminance(np.eye(3, dtype=np.float32).reshape(1, 3, 3)),
        [[0.2126, 0.7152, 0.0722]],
    )


def test_blur_matches_gaussian_impulse_and_preserves_interior_flux(cfg: Config) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 1.0
    obs = cfg["observation"]
    blurred = gaussian_blur(image, obs["sigma_psf"], obs["kernel_size"])
    yy, xx = np.mgrid[-3:4, -3:4]
    expected = np.exp(-(xx**2 + yy**2) / 2)
    expected /= expected.sum()
    np.testing.assert_allclose(blurred[29:36, 29:36], expected, rtol=1e-6)
    assert blurred.sum(dtype=np.float64) == pytest.approx(1.0, rel=1e-4)
    assert np.count_nonzero(blurred) == 49


def test_zero_padding_loses_flux_at_boundary(cfg: Config) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[0, 0] = 1.0
    blurred = gaussian_blur(
        image, cfg["observation"]["sigma_psf"], cfg["observation"]["kernel_size"]
    )
    assert 0 < blurred.sum() < 1
    assert np.all(blurred[-1] == 0)
    assert np.all(blurred[:, -1] == 0)


def test_preprocessing_normalizes_flux_and_discards_brightness(cfg: Config) -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[20:40, 20:40] = [1.0, 2.0, 3.0]
    processed = preprocess_image(rgb, cfg)
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32
    assert processed.sum(dtype=np.float64) == pytest.approx(4096, rel=1e-7)
    np.testing.assert_allclose(preprocess_image(rgb * 10, cfg), processed, rtol=1e-6)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_preprocessing_rejects_invalid_radiance(cfg: Config, value: float) -> None:
    with pytest.raises(ValueError):
        preprocess_image(np.full((64, 64, 3), value, dtype=np.float32), cfg)


def test_noise_is_seeded_fresh_and_gaussian() -> None:
    x = jnp.zeros((32, 64, 64), dtype=jnp.float32)
    noise = add_noise(x, jax.random.key(1), sigma_n=0.3)
    np.testing.assert_array_equal(noise, add_noise(x, jax.random.key(1), sigma_n=0.3))
    assert not np.array_equal(noise, add_noise(x, jax.random.key(2), sigma_n=0.3))
    assert float(noise.mean()) == pytest.approx(0.0, abs=0.004)
    assert float(noise.std()) == pytest.approx(0.3, rel=0.01)
    assert np.any(np.asarray(noise) < 0)


def test_observation_is_jittable_on_cpu_and_asinh_is_invertible() -> None:
    x = jnp.linspace(-10, 10, 64 * 64).reshape(64, 64)
    transformed = jax.jit(lambda x, key: observe(x, key, sigma_n=0.0, s=0.7))(x, jax.random.key(1))
    np.testing.assert_allclose(0.7 * jnp.sinh(transformed), x, rtol=1e-6, atol=1e-6)
    assert transformed.shape == x.shape
    assert all(device.platform == "cpu" for device in jax.devices())


@pytest.mark.parametrize("s", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_asinh_scale(s: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        network_transform(jnp.zeros((1,)), s)
