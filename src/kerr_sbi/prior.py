import numpy as np

from kerr_sbi.config import Config


def sample_prior(n: int, seed: int, cfg: Config) -> np.ndarray:
    if type(n) is not int or n < 0 or type(seed) is not int or seed < 0:
        raise ValueError("sample count and seed must be nonnegative integers")
    prior = cfg["prior"]
    limits = [prior[key] for key in ("a_min", "a_max", "incl_min_deg", "incl_max_deg")]
    if not np.isfinite(limits).all():
        raise ValueError("prior limits must be finite")
    if not 0 <= limits[0] < limits[1] <= 0.98 or not 0 < limits[2] < limits[3] < 90:
        raise ValueError("invalid v1 prior limits")
    draws = np.random.Generator(np.random.PCG64(seed)).random((n, 2))
    a = prior["a_min"] + draws[:, 0] * (prior["a_max"] - prior["a_min"])
    c_lo, c_hi = np.cos(np.deg2rad([prior["incl_max_deg"], prior["incl_min_deg"]]))
    cos_i = c_lo + draws[:, 1] * (c_hi - c_lo)
    return np.column_stack([a, np.rad2deg(np.arccos(cos_i)), cos_i])
