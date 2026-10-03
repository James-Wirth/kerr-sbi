import numpy as np
import pytest

from kerr_sbi.visibility import (
    UAS_TO_RAD,
    FourierImage,
    VisibilityData,
    direct_visibility,
    fit_station_gains,
    fit_visibility_pair,
    load_eht_csv,
    matched_flux,
    visibility_distance,
)


def test_centered_pixel_and_symmetric_pair() -> None:
    uv = np.array([[0, 0], [2e9, 3e9], [-4e9, 1e9]])
    image = np.zeros((7, 7))
    image[3, 3] = 1
    pixel = 2 * UAS_TO_RAD
    expected = np.sinc(uv * pixel).prod(axis=1)
    np.testing.assert_allclose(direct_visibility(image, uv, 2), expected, atol=1e-14)
    image[3, 3] = 0
    image[3, 1] = image[3, 5] = 1
    expected *= np.cos(2 * np.pi * uv[:, 0] * 2 * pixel)
    np.testing.assert_allclose(direct_visibility(image, uv, 2), expected, atol=1e-14)


@pytest.mark.parametrize("shape", [(8, 8), (9, 7), (64, 64)])
def test_fourier_interpolation_against_direct_sum(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(91)
    image = rng.uniform(size=shape)
    uv = rng.uniform(-8e9, 8e9, (100, 2))
    model = FourierImage(image)
    exact = direct_visibility(image, uv, 2, 31, (3, -5))
    np.testing.assert_allclose(model.evaluate(uv, 2, 31, (3, -5)), exact, atol=3e-6)
    np.testing.assert_allclose(model.evaluate(np.zeros((1, 2)), 2), 1, atol=1e-14)
    np.testing.assert_allclose(model.evaluate(-uv, 2), model.evaluate(uv, 2).conj(), atol=3e-6)


def test_sky_orientation_and_translation() -> None:
    image = np.zeros((9, 9))
    image[3, 6] = 1
    uv = np.array([[2e9, 3e9], [-4e9, 1e9]])
    aperture = np.sinc(uv * 2 * UAS_TO_RAD).prod(axis=1)
    expected = aperture * np.exp(-2j * np.pi * (uv @ [4, 2]) * UAS_TO_RAD)
    np.testing.assert_allclose(direct_visibility(image, uv, 2), expected, atol=1e-14)
    expected_rotated = aperture * np.exp(-2j * np.pi * (uv @ [-2, 4]) * UAS_TO_RAD)
    np.testing.assert_allclose(direct_visibility(image, uv, 2, 90), expected_rotated, atol=1e-14)
    shifted = direct_visibility(image, uv, 2, offset_uas=(-4, -2))
    np.testing.assert_allclose(shifted, aperture, atol=1e-14)


def test_flux_and_gaussian_distance() -> None:
    model = np.array([1 + 2j, 2 - 1j])
    sigma = np.array([0.1, 0.2])
    assert matched_flux(model, 0.6 * model, sigma, (0.3, 1.2)) == pytest.approx(0.6)
    assert matched_flux(model, 2 * model, sigma, (0.3, 1.2)) == 1.2
    assert visibility_distance(model, model + sigma * (1 + 1j), sigma) == pytest.approx(2)
    with pytest.raises(ValueError):
        visibility_distance(model, model, -sigma)


def test_pair_fit_recovers_known_transform() -> None:
    rng = np.random.default_rng(49)
    image = np.zeros((16, 16))
    image[3:7, 2:5] = 2
    image[8:11, 9:14] = 1
    uv = rng.uniform(-8e9, 8e9, (80, 2))
    model = FourierImage(image)
    target = 0.7 * direct_visibility(image, uv, 2.2, 17, (2, -3))
    fit = fit_visibility_pair(
        model,
        target,
        uv,
        np.full(80, 0.01),
        2,
        np.array([[0.75, -90, -20, -20], [1.25, 90, 20, 20]]),
        (0.3, 1.2),
        [[1, 0, 0, 0]],
        100,
    )
    assert fit["optimizer_success"]
    assert fit["distance"] < 0.001
    np.testing.assert_allclose(fit["parameters"], [1.1, 17, 2, -3], atol=1e-3)
    assert fit["flux_jy"] == pytest.approx(0.7, abs=1e-5)


def test_csv_selection_and_errors(tmp_path) -> None:
    path = tmp_path / "eht.csv"
    header = "#SRC:M87,DATE(MJD):57854,FREQ:227.0707GHz\n"
    columns = "#time(UTC),T1,T2,U(lambda),V(lambda),Iamp(Jy),Iphase(d),Isigma(Jy)\n"
    rows = "1,AA,AP,100,200,0.6,90,0.01\n2,AA,PV,3e9,4e9,0.1,0,0.02\n"
    path.write_text(header + columns + rows)
    data = load_eht_csv(path, 1e8)
    assert data.uv.shape == (1, 2)
    assert data.observed[0] == pytest.approx(0.1)
    assert data.sigma[0] == 0.02
    path.write_text(header + columns + rows.replace("0.02", "-0.02"))
    with pytest.raises(ValueError, match="invalid"):
        load_eht_csv(path)


def test_station_gain_recovery_and_closure_invariance() -> None:
    stations = np.array([["A", "B"], ["B", "C"], ["A", "C"]] * 3)
    candidate = np.array([1 + 0.1j, 0.6 + 0.4j, 0.3 - 0.2j] * 3)
    gains = {"A": 1.03, "B": 0.98 * np.exp(0.2j), "C": 1.01 * np.exp(-0.1j)}
    target = candidate * np.array([gains[a] * np.conj(gains[b]) for a, b in stations])
    data = VisibilityData(
        np.zeros(9), stations, np.zeros((9, 2)), target, np.full(9, 0.01), "synthetic"
    )
    prediction, result = fit_station_gains(target, candidate, data, 300, (0.8, 1.2))
    assert result["distance"] < 1e-7
    assert result["all_optimizers_succeeded"]

    def phase(v: np.ndarray) -> float:
        return float(np.angle(v[0] * v[1] * v[2].conj()))

    assert phase(prediction) == pytest.approx(phase(candidate), abs=1e-12)
