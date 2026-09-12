import numpy as np
import pytest

from kerr_sbi.config import Config
from kerr_sbi.prior import sample_prior


def test_prior_extension_preserves_parameter_pairs(cfg: Config) -> None:
    small = sample_prior(30, 1, cfg)
    large = sample_prior(12000, 1, cfg)
    np.testing.assert_array_equal(small, large[:30])
    assert not np.array_equal(small, sample_prior(30, 2, cfg))
    np.testing.assert_allclose(np.cos(np.deg2rad(large[:, 1])), large[:, 2], atol=2e-16)


def test_prior_is_uniform_in_spin_and_cosine(cfg: Config) -> None:
    parameters = sample_prior(20000, 71, cfg)
    prior = cfg["prior"]
    assert np.all(
        (parameters[:, 1] >= prior["incl_min_deg"]) & (parameters[:, 1] <= prior["incl_max_deg"])
    )
    c_lo, c_hi = np.cos(np.deg2rad([prior["incl_max_deg"], prior["incl_min_deg"]]))
    uniform = np.column_stack(
        [
            (parameters[:, 0] - prior["a_min"]) / (prior["a_max"] - prior["a_min"]),
            (parameters[:, 2] - c_lo) / (c_hi - c_lo),
        ]
    )
    assert np.all((uniform >= 0) & (uniform <= 1))
    for values in uniform.T:
        counts, _ = np.histogram(values, bins=np.linspace(0, 1, 6))
        assert np.max(np.abs(counts - 4000)) < 250
    assert abs(np.corrcoef(uniform.T)[0, 1]) < 0.03


@pytest.mark.parametrize(("n", "seed"), [(-1, 1), (1, -1), (1.5, 1), (1, True)])
def test_prior_rejects_invalid_count_or_seed(cfg: Config, n: int, seed: int) -> None:
    with pytest.raises(ValueError):
        sample_prior(n, seed, cfg)
