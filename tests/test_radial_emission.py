import numpy as np
import pytest
from test_transfer import fixture_transfer

from kerr_sbi.radial_emission import RadialFourier, radial_basis, radial_intensity
from kerr_sbi.transfer import Transfer
from kerr_sbi.transfer_fourier import TransferFourier
from kerr_sbi.visibility import UAS_TO_RAD


@pytest.mark.parametrize("jitter", [False, True])
def test_radial_moments_match_independent_sample_quadrature(tmp_path, jitter):
    original = fixture_transfer(tmp_path, 2, 4, jitter)
    samples = original.samples.copy()
    samples["radius"] = np.linspace(8, 15, len(samples))
    samples["g"] = np.linspace(0.5, 1.2, len(samples))
    transfer = Transfer(samples, original.metadata, original.scene)
    coefficients = np.array([1, 0.25, 4, 0.7])
    t = np.log(samples["radius"] / 8) / np.log(15 / 8)
    modifier = (1 - t) ** 3 + 0.75 * t * (1 - t) ** 2 + 12 * t**2 * (1 - t) + 0.7 * t**3
    intensity = samples["g"] ** 3 * (samples["radius"] / 8) ** -2.04 * modifier
    np.testing.assert_allclose(radial_intensity(transfer, 2.04, coefficients), intensity)
    radial = RadialFourier(transfer, 2.04)
    model = radial.combine(coefficients)
    uv = np.array([[0, 0], [3e9, -4e9], [-3e9, 4e9], [12e9, 0]])
    angle = np.deg2rad(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = np.column_stack((samples["screen_u"], samples["screen_v"]))
    xy *= 6 * UAS_TO_RAD / (2 * np.tan(np.deg2rad(22) / 2))
    xy = xy @ rotation.T + np.array([0.7, -0.2]) * UAS_TO_RAD
    masses = intensity * samples["weight"]
    expected = np.exp(-2j * np.pi * (uv @ xy.T)) @ (0.6 * masses / masses.sum())
    kwargs = dict(fov_uas=6, flux_jy=0.6, angle_deg=37, offset_uas=(0.7, -0.2))
    actual, bound = model.evaluate(uv, **kwargs)
    direct = transfer.visibility(uv, values=intensity, sample_batch=7, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(direct, expected, atol=2e-15)
    assert np.max(bound) < 1e-12
    unity = radial.combine(np.ones(4))
    ordinary = TransferFourier(transfer, 2.04, 3)
    np.testing.assert_allclose(unity.moments, ordinary.moments, atol=1e-15)
    np.testing.assert_allclose(radial_basis(np.linspace(8, 15, 101), 8, 15).sum(axis=1), 1)


def test_custom_emission_preserves_support_and_rejects_failures(tmp_path):
    transfer = fixture_transfer(tmp_path)
    samples = transfer.samples.copy()
    samples["g_available"][0] = samples["g_valid"][0] = 0
    samples["g"][0] = np.nan
    changed = Transfer(samples, transfer.metadata, transfer.scene)
    values = radial_intensity(changed, 2, np.ones(4))
    assert values[0] == 0
    values[0] = 1
    with pytest.raises(ValueError, match="support"):
        TransferFourier(changed, values=values)
    with pytest.raises(ValueError, match="ambiguous"):
        transfer.visibility(np.zeros((1, 2)), 125, emissivity_index=2, values=values)
    with pytest.raises(ValueError, match="invalid sample"):
        TransferFourier(transfer, values=-values)
    samples["finished"][1] = 0
    with pytest.raises(ValueError, match="unfinished"):
        RadialFourier(changed, 2)
    with pytest.raises(ValueError, match="fixed emitting edges"):
        radial_basis(np.array([7.99]), 8, 15)
