import numpy as np
import pytest

from kerr_sbi.fisher import central_difference, fisher_bounds


def test_central_difference_quadratic_and_degree_units() -> None:
    a, inclination = 0.4, 35.0

    def image(spin: float, degrees: float) -> np.ndarray:
        return np.array([spin**2 + degrees, 3 * spin + degrees**2])

    np.testing.assert_allclose(
        central_difference(image(a - 0.02, inclination), image(a + 0.02, inclination), 0.02),
        [2 * a, 3],
    )
    np.testing.assert_allclose(
        central_difference(image(a, inclination - 1), image(a, inclination + 1), 1),
        [1, 2 * inclination],
    )


def test_difference_promotes_before_subtraction() -> None:
    minus = np.array([-3e38], dtype=np.float32)
    plus = -minus
    result = central_difference(minus, plus, 1)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, plus.astype(np.float64))


def test_independent_parameters() -> None:
    result = fisher_bounds(np.diag([1.0, 2.0]), 0.2)
    np.testing.assert_allclose(result["fisher"], np.diag([25, 100]))
    np.testing.assert_allclose(result["marginal_std"], [0.2, 0.1])
    np.testing.assert_array_equal(result["marginal_std"], result["conditional_std"])
    assert result["correlation"] == 0
    assert not result["singular"]


def test_correlated_parameters_and_batched_noise_scaling() -> None:
    jacobian = np.broadcast_to([[2.0, 1.0], [0.0, 1.0]], (3, 4, 2, 2))
    low = fisher_bounds(jacobian, 0.2)
    high = fisher_bounds(jacobian, 0.6)
    np.testing.assert_allclose(low["marginal_std"][0, 0], 0.2 * np.sqrt([0.5, 1]))
    np.testing.assert_allclose(low["conditional_std"][0, 0], 0.2 / np.sqrt([4, 2]))
    np.testing.assert_allclose(low["correlation"], -1 / np.sqrt(2))
    np.testing.assert_allclose(high["marginal_std"], 3 * low["marginal_std"])
    np.testing.assert_allclose(high["fisher"], low["fisher"] / 9)
    covariance = np.linalg.inv(low["fisher"])
    np.testing.assert_allclose(
        low["marginal_std"], np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    )


@pytest.mark.parametrize("jacobian", [[[1, 1], [2, 2]], [[0, 1], [0, 2]], [[0, 0]]])
def test_singular_information_has_no_finite_joint_bound(jacobian: list) -> None:
    result = fisher_bounds(np.array(jacobian), 1)
    assert result["singular"]
    assert np.all(np.isinf(result["marginal_std"]))
    assert np.isnan(result["correlation"])


def test_singularity_check_is_invariant_to_parameter_units() -> None:
    jacobian = np.array([[2.0, 1.0], [0.0, 1.0]])
    original = fisher_bounds(jacobian, 1)
    scaled = fisher_bounds(jacobian * [1, 1e-8], 1)
    assert not scaled["singular"]
    np.testing.assert_allclose(scaled["correlation"], original["correlation"])
    np.testing.assert_allclose(scaled["marginal_std"], original["marginal_std"] * [1, 1e8])


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf])
def test_invalid_noise(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma_n"):
        fisher_bounds(np.eye(2), sigma)


@pytest.mark.parametrize("jacobian", [[], [1, 2], [[1, 2, 3]], [[1, np.nan]], np.empty((0, 2))])
def test_invalid_jacobian(jacobian: list | np.ndarray) -> None:
    with pytest.raises(ValueError, match="Jacobian"):
        fisher_bounds(np.array(jacobian), 1)


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf])
def test_invalid_difference_step(step: float) -> None:
    with pytest.raises(ValueError, match="step"):
        central_difference(np.ones(2), np.ones(2), step)


def test_invalid_difference_arrays() -> None:
    for minus, plus in [([], []), ([1], [1, 2]), ([np.nan], [1])]:
        with pytest.raises(ValueError, match="arrays"):
            central_difference(np.array(minus), np.array(plus), 1)
