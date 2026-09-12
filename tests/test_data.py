from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerr_sbi.config import Config
from kerr_sbi.data import (
    batch_order,
    fit_normalization,
    load_split,
    prior_log_prob,
    theta_to_u,
    training_observations,
    training_rows,
    u_to_theta,
    validation_observations,
)
from kerr_sbi.dummy import build_dummy_dataset, dummy_config


@pytest.fixture
def training_data(cfg: Config, tmp_path: Path) -> tuple:
    cfg["project_root"] = tmp_path
    build_dummy_dataset(cfg)
    cfg = dummy_config(cfg)
    data = load_split(cfg, "train", allow_dummy=True)
    train, validation = training_rows(data, 0.1)
    return cfg, data, train, validation, fit_normalization(data, train, cfg)


def test_loading_requires_matching_mode_and_observation_model(training_data: tuple) -> None:
    cfg, data, train, validation, _ = training_data
    assert len(train) == 57 and len(validation) == 7
    with pytest.raises(ValueError, match="mode"):
        load_split(cfg, "train")
    changed = deepcopy(cfg)
    changed["observation"]["sigma_n"] = 1.0
    with pytest.raises(ValueError, match="observation"):
        load_split(changed, "train", allow_dummy=True)
    assert data.x.shape == (64, 64, 64)


def test_normalization_excludes_validation_and_test(training_data: tuple) -> None:
    cfg, data, train, validation, stats = training_data
    changed = replace(data, x=data.x.copy(), theta=data.theta.copy())
    changed.x[validation] *= 100
    changed.theta[validation] = [0.97, 79]
    assert fit_normalization(changed, train, cfg) == stats
    with pytest.raises(ValueError, match="training-only"):
        fit_normalization(data, np.arange(64), cfg)
    test = load_split(cfg, "test", allow_dummy=True)
    with pytest.raises(ValueError, match="training split"):
        fit_normalization(test, np.arange(16), cfg)


def test_partial_data_uses_original_validation_boundary(training_data: tuple) -> None:
    _, data, _, _, _ = training_data
    rows = np.r_[np.arange(10, 20), np.arange(57, 64)]
    partial = replace(data, x=data.x[rows], theta=data.theta[rows], idx=data.idx[rows])
    train, val = training_rows(partial, 0.1)
    np.testing.assert_array_equal(partial.idx[val], np.arange(57, 64))
    assert len(train) == 10
    partial = replace(partial, idx=partial.idx[:-1])
    with pytest.raises(ValueError, match="incomplete validation"):
        training_rows(partial, 0.1)


def test_parameter_round_trip_and_clipping(training_data: tuple) -> None:
    _, data, train, _, stats = training_data
    theta = jnp.asarray(data.theta[train])
    actual = jax.jit(lambda t: u_to_theta(theta_to_u(t, stats), stats))(theta)
    np.testing.assert_allclose(actual, theta, atol=1e-4, rtol=1e-5)
    edges = jnp.array([[0.0, 80.0], [0.98, 5.0]])
    clipped = u_to_theta(theta_to_u(edges, stats), stats)
    assert np.isfinite(clipped).all()
    assert 0 < clipped[0, 0] < clipped[1, 0] < 0.98


def test_prior_log_density_matches_inverse_transform_jacobian(training_data: tuple) -> None:
    _, _, _, _, stats = training_data
    u = jnp.array([0.3, -0.6])
    theta = u_to_theta(u, stats)
    jacobian = jax.jacfwd(lambda v: u_to_theta(v, stats))(u)
    prior_theta = jnp.sin(jnp.deg2rad(theta[1])) * jnp.pi / 180
    prior_theta /= (stats.upper[0] - stats.lower[0]) * (stats.upper[1] - stats.lower[1])
    expected = jnp.log(prior_theta) + jnp.linalg.slogdet(jacobian)[1]
    np.testing.assert_allclose(prior_log_prob(u, stats), expected, atol=2e-6)


def test_batch_masks_cover_small_and_tail_batches_once() -> None:
    for n in (3, 17):
        order, mask = batch_order(n, 8, jax.random.PRNGKey(1))
        assert order.shape[1] == 8
        np.testing.assert_array_equal(np.sort(np.asarray(order)[mask]), np.arange(n))
        np.testing.assert_array_equal(order, batch_order(n, 8, jax.random.PRNGKey(1))[0])


def test_training_noise_and_index_stable_validation_noise(training_data: tuple) -> None:
    _, data, _, val, stats = training_data
    x, idx = jnp.asarray(data.x[val]), jnp.asarray(data.idx[val])
    first = training_observations(x, jax.random.PRNGKey(1), stats, 20.0)
    assert first.shape == (7, 1, 64, 64) and np.any(np.asarray(first) < 0)
    np.testing.assert_array_equal(first, training_observations(x, jax.random.PRNGKey(1), stats, 20))
    assert not np.array_equal(first, training_observations(x, jax.random.PRNGKey(2), stats, 20))
    fixed = validation_observations(x, idx, 42, stats, 20)
    reversed_batch = validation_observations(x[::-1], idx[::-1], 42, stats, 20)
    np.testing.assert_array_equal(fixed, reversed_batch[::-1])
    np.testing.assert_array_equal(fixed[:2], validation_observations(x[:2], idx[:2], 42, stats, 20))
