import numpy as np
import pytest

from kerr_sbi.luminet import (
    corrected_redshift,
    first_crossing_angle,
    impact_coordinates,
    orbit_radius,
    reconstruct_rings,
    redshift,
    ring_reference,
)
from kerr_sbi.transfer import SAMPLE_DTYPE


def test_local_impact_and_energy_conversion():
    radius = 100.0
    delta = 0.2
    xy = impact_coordinates(np.array([np.tan(delta)]), np.array([0.0]), radius)
    assert xy[0, 0] == pytest.approx(radius * np.sin(delta) / np.sqrt(0.98))
    assert corrected_redshift(np.array([1 / np.sqrt(0.98)]), radius)[0] == pytest.approx(0)
    with pytest.raises(ValueError):
        impact_coordinates(np.array([0.0]), np.array([0.0]), 2)


def test_plane_angles_and_redshift_orientation():
    angles = first_crossing_angle(np.array([-np.pi / 2, 0, np.pi / 2]), 60)
    np.testing.assert_allclose(angles, np.deg2rad([30, 90, 150]))
    assert redshift(np.array([6.0]), np.array([0.0]), 60)[0] == pytest.approx(np.sqrt(2) - 1)
    assert redshift(np.array([10.0]), np.array([-10.0]), 60)[0] < 0
    assert redshift(np.array([10.0]), np.array([10.0]), 60)[0] > 0


def test_orbit_against_independent_radial_integral():
    from scipy.integrate import quad

    impact, radius = 8.0, 12.0
    angle = quad(lambda u: 1 / np.sqrt(1 / impact**2 - u**2 + 2 * u**3), 0, 1 / radius)[0]
    assert orbit_radius(impact, angle, np.inf, 2e-12) == pytest.approx(radius, abs=1e-8)
    assert orbit_radius(1.0, 2.0, np.inf, 2e-12) == 2


def test_reference_symmetry_and_shooting():
    beta = np.array([0.0, np.pi])
    b = ring_reference(np.array([10.0]), beta, 60, np.inf, 2e-11)
    np.testing.assert_allclose(b[0, 0], b[0, 1], atol=1e-10)
    assert orbit_radius(b[0, 0], np.pi / 2, np.inf, 2e-13) == pytest.approx(10, abs=1e-7)


def test_stencil_retains_invalid_nodes():
    samples = np.zeros(25, dtype=SAMPLE_DTYPE)
    x, y = np.meshgrid(np.linspace(0.08, 0.12, 5), np.linspace(-0.02, 0.02, 5))
    samples["screen_u"], samples["screen_v"] = x.ravel(), y.ravel()
    xy = impact_coordinates(samples["screen_u"], samples["screen_v"], 100)
    samples["radius"] = xy[:, 0]
    samples["g"] = 1.0
    for name in ["has_intersection", "radius_valid", "g_valid", "g_available", "finished"]:
        samples[name] = 1
    args = (samples, 100, np.array([0.0]), np.array([10.0]), np.array([[10.0]]), 12, 3, 5, 0.5)
    result = reconstruct_rings(*args)
    assert result["valid"][0, 0]
    assert result["b"][0, 0] == pytest.approx(10)
    samples["finished"] = 0
    assert not reconstruct_rings(*args)["valid"][0, 0]
