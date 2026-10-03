import numpy as np
import pytest
from test_transfer import fixture_transfer

from kerr_sbi.transfer import Transfer
from kerr_sbi.transfer_fourier import TransferFourier


@pytest.mark.parametrize("n,nmax,jitter", [(3, 3, False), (3, 3, True), (1, 4, True)])
def test_moments_against_direct_with_transforms_and_raw_flux(tmp_path, n, nmax, jitter):
    transfer = fixture_transfer(tmp_path, n, nmax, jitter)
    samples = transfer.samples.copy()
    samples["g"] = np.linspace(0.1, 1.3, len(samples))
    transfer = Transfer(samples, transfer.metadata, transfer.scene)
    fourier = TransferFourier(transfer, sample_batch=7)
    uv = np.array([[0, 0], [2e9, -5e9], [-2e9, 5e9], [9e9, 1e9]])
    for flux in (None, 0.6):
        kwargs = dict(fov_uas=6.0, flux_jy=flux, angle_deg=37.0, offset_uas=(3.1, -2.7))
        expected = transfer.visibility(uv, **kwargs)
        actual, bound = fourier.evaluate(uv, baseline_batch=1, **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-35)
        assert np.max(bound) < 1e-12
        mask = np.arange(6) < 3
        a, _ = fourier.evaluate(uv, pixel_mask=mask, **kwargs)
        b, _ = fourier.evaluate(uv, pixel_mask=~mask, **kwargs)
        np.testing.assert_allclose(a + b, expected, rtol=1e-14, atol=1e-35)


def test_truncation_bound_controls_actual_error_and_refuses_large_phase(tmp_path):
    transfer = fixture_transfer(tmp_path, 3, 3, True)
    uv = np.array([[0, 0], [2e9, 3e9], [-8e9, 5e9]])
    fourier = TransferFourier(transfer, order=2)
    actual, bound = fourier.evaluate(uv, 20, flux_jy=0.6, max_truncation_error=10)
    expected = transfer.visibility(uv, 20, flux_jy=0.6)
    assert np.all(np.abs(actual - expected) <= bound + 1e-15)
    assert np.max(np.abs(actual - expected)) > 1e-4
    with pytest.raises(ValueError, match="allowance exceeded"):
        fourier.evaluate(uv, 20, flux_jy=0.6)


def test_moments_reject_unfinished_and_invalid_emission(tmp_path):
    transfer = fixture_transfer(tmp_path)
    s = transfer.samples.copy()
    s["finished"][0] = 0
    with pytest.raises(ValueError, match="unfinished"):
        TransferFourier(Transfer(s, transfer.metadata, transfer.scene))
    s["finished"][0] = 1
    s["g_valid"][0] = 0
    with pytest.raises(ValueError, match="invalid emission"):
        TransferFourier(Transfer(s, transfer.metadata, transfer.scene))
