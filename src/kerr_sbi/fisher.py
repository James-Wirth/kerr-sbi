import math

import numpy as np


def central_difference(minus: np.ndarray, plus: np.ndarray, step: float) -> np.ndarray:
    if not math.isfinite(step) or step <= 0:
        raise ValueError("difference step must be finite and positive")
    minus = np.asarray(minus, dtype=np.float64)
    plus = np.asarray(plus, dtype=np.float64)
    if minus.shape != plus.shape or minus.size == 0:
        raise ValueError("difference arrays must have matching nonempty shapes")
    if not np.all(np.isfinite(minus)) or not np.all(np.isfinite(plus)):
        raise ValueError("difference arrays must be finite")
    return (plus - minus) / (2 * step)


def fisher_bounds(
    jacobian: np.ndarray, sigma_n: float, singular_tolerance: float = 1e-12
) -> dict[str, np.ndarray]:
    if not math.isfinite(sigma_n) or sigma_n <= 0:
        raise ValueError("sigma_n must be finite and positive")
    if not math.isfinite(singular_tolerance) or not 0 < singular_tolerance < 1:
        raise ValueError("singular_tolerance must lie between zero and one")
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim < 2 or jacobian.shape[-1] != 2 or jacobian.shape[-2] == 0:
        raise ValueError("expected Jacobian shape (..., pixels, 2)")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian must be finite")
    gram = np.einsum("...pi,...pj->...ij", jacobian, jacobian)
    norms = np.sqrt(np.diagonal(gram, axis1=-2, axis2=-1))
    conditional = np.full_like(norms, np.inf)
    np.divide(sigma_n, norms, out=conditional, where=norms > 0)
    norm_product = norms[..., 0] * norms[..., 1]
    cosine = np.zeros_like(norm_product)
    np.divide(gram[..., 0, 1], norm_product, out=cosine, where=norm_product > 0)
    cosine = np.clip(cosine, -1, 1)
    residual_fraction = 1 - cosine**2
    singular = (norm_product == 0) | (residual_fraction <= singular_tolerance)
    marginal = np.full_like(conditional, np.inf)
    np.divide(
        conditional,
        np.sqrt(residual_fraction)[..., None],
        out=marginal,
        where=~singular[..., None],
    )
    return {
        "fisher": gram / sigma_n**2,
        "marginal_std": marginal,
        "conditional_std": conditional,
        "correlation": np.where(singular, np.nan, -cosine),
        "singular": singular,
    }
