from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from kerr_sbi.config import Config, project_path
from kerr_sbi.dataset import dataset_identity, file_sha256, load_preprocessed, read_parameters
from kerr_sbi.obs_model import observe
from kerr_sbi.prior import sample_prior


@dataclass(frozen=True)
class SplitData:
    x: np.ndarray
    theta: np.ndarray
    idx: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Normalization:
    mean: tuple[float, float]
    std: tuple[float, float]
    lower: tuple[float, float]
    upper: tuple[float, float]
    clip: float
    asinh_scale: float


def load_split(cfg: Config, split: str, *, allow_dummy: bool = False) -> SplitData:
    arrays, meta = load_preprocessed(cfg, split)
    is_dummy = meta.get("dataset_kind") == "dummy"
    if is_dummy != allow_dummy:
        raise ValueError("dataset kind differs from the explicitly selected dummy/production mode")
    if meta["split"] != split or meta["theta_columns"] != ["a", "incl_deg"]:
        raise ValueError("dataset split or parameter columns differ")
    if meta["prior"] != cfg["prior"] or meta["observation"] != cfg["observation"]:
        raise ValueError("dataset prior or observation configuration differs")
    if is_dummy:
        if meta.get("scientific_use") is not False or meta["nullgeo"] is not None:
            raise ValueError("invalid dummy provenance")
    elif meta["identity"] != dataset_identity(cfg, meta["nullgeo"]):
        raise ValueError("production dataset identity differs")
    x, theta, idx = arrays["x"], arrays["theta"], arrays["idx"]
    n = len(idx)
    if (
        x.shape != (n, 64, 64)
        or x.dtype != np.float32
        or theta.shape != (n, 2)
        or theta.dtype != np.float64
        or idx.shape != (n,)
        or idx.dtype != np.int64
    ):
        raise ValueError("unexpected dataset array shape or dtype")
    if not np.isfinite(x).all() or np.any(x < 0) or not np.isfinite(theta).all():
        raise ValueError("nonfinite parameters or invalid preprocessed images")
    if not np.allclose(x.sum(axis=(1, 2), dtype=np.float64), cfg["observation"]["flux_sum"]):
        raise ValueError("preprocessed image flux differs")
    requested = meta["n_requested"]
    if (
        type(requested) is not int
        or requested < 1
        or np.any(idx < 0)
        or np.any(idx >= requested)
        or np.any(np.diff(idx) <= 0)
    ):
        raise ValueError("invalid or unordered original dataset indices")
    params_path = project_path(cfg, "data") / split / "params.csv"
    if file_sha256(params_path) != meta["params_sha256"]:
        raise ValueError("parameter table changed; rerun preprocessing")
    params = read_parameters(params_path)
    if not np.array_equal(params, sample_prior(requested, meta["seed"], cfg)):
        raise ValueError("parameter table differs from the seeded prior")
    if not np.array_equal(theta, params[idx, :2]):
        raise ValueError("image parameter/index alignment differs")
    if n != meta["n_examples" if is_dummy else "n_rendered"]:
        raise ValueError("dataset example count differs")
    return SplitData(x, theta, idx, meta)


def training_rows(data: SplitData, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1 or data.metadata["split"] != "train":
        raise ValueError("training split and a validation fraction between zero and one required")
    boundary = int(np.floor((1 - validation_fraction) * data.metadata["n_requested"]))
    train = np.flatnonzero(data.idx < boundary)
    validation = np.flatnonzero(data.idx >= boundary)
    if len(train) < 2 or len(validation) < 1:
        raise ValueError("at least two training rows and one validation row required")
    if len(validation) != data.metadata["n_requested"] - boundary:
        raise ValueError("incomplete validation coverage at the original index boundary")
    return train, validation


def fit_normalization(data: SplitData, rows: np.ndarray, cfg: Config) -> Normalization:
    allowed, _ = training_rows(data, cfg["training"]["validation_fraction"])
    rows = np.asarray(rows)
    if rows.ndim != 1 or len(rows) < 2 or not np.isin(rows, allowed).all():
        raise ValueError("normalization requires training-only rows")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("normalization rows must be unique")
    prior = cfg["prior"]
    lower = np.array([prior["a_min"], np.cos(np.deg2rad(prior["incl_max_deg"]))])
    upper = np.array([prior["a_max"], np.cos(np.deg2rad(prior["incl_min_deg"]))])
    clip = prior["logit_clip"]
    if not np.isfinite(clip) or not 0 < clip < 0.5:
        raise ValueError("logit clipping must be between zero and one half")
    theta = data.theta[rows]
    physical = np.column_stack([theta[:, 0], np.cos(np.deg2rad(theta[:, 1]))])
    unit = np.clip((physical - lower) / (upper - lower), clip, 1 - clip)
    logits = np.log(unit) - np.log1p(-unit)
    mean, std = logits.mean(axis=0), logits.std(axis=0)
    obs = cfg["observation"]
    scale = obs["s"]
    if scale == "training_median":
        scale = obs["asinh_scale_factor"] * np.median(data.x[rows].max(axis=(1, 2)))
    values = np.r_[mean, std, scale]
    if not np.isfinite(values).all() or np.any(std <= 0) or scale <= 0:
        raise ValueError("normalization statistics must have finite positive scales")
    return Normalization(tuple(mean), tuple(std), tuple(lower), tuple(upper), clip, float(scale))


def theta_to_u(theta: jax.Array, stats: Normalization) -> jax.Array:
    theta = jnp.asarray(theta)
    physical = jnp.stack([theta[..., 0], jnp.cos(jnp.deg2rad(theta[..., 1]))], axis=-1)
    unit = (physical - jnp.array(stats.lower)) / (jnp.array(stats.upper) - jnp.array(stats.lower))
    unit = jnp.clip(unit, stats.clip, 1 - stats.clip)
    logits = jnp.log(unit) - jnp.log1p(-unit)
    return (logits - jnp.array(stats.mean)) / jnp.array(stats.std)


def u_to_theta(u: jax.Array, stats: Normalization) -> jax.Array:
    unit = jax.nn.sigmoid(jnp.asarray(u) * jnp.array(stats.std) + jnp.array(stats.mean))
    physical = jnp.array(stats.lower) + unit * (jnp.array(stats.upper) - jnp.array(stats.lower))
    return jnp.stack([physical[..., 0], jnp.rad2deg(jnp.arccos(physical[..., 1]))], axis=-1)


def prior_log_prob(u: jax.Array, stats: Normalization) -> jax.Array:
    logits = jnp.asarray(u) * jnp.array(stats.std) + jnp.array(stats.mean)
    return jnp.sum(
        -jax.nn.softplus(-logits) - jax.nn.softplus(logits) + jnp.log(jnp.array(stats.std)), axis=-1
    )


def batch_order(n: int, batch_size: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    if n < 1 or batch_size < 1:
        raise ValueError("batching requires positive sample and batch counts")
    padded = ((n + batch_size - 1) // batch_size) * batch_size
    order = jax.random.permutation(key, n)
    order = jnp.pad(order, (0, padded - n), mode="edge")
    return order.reshape(-1, batch_size), (jnp.arange(padded) < n).reshape(-1, batch_size)


def training_observations(
    x: jax.Array, key: jax.Array, stats: Normalization, sigma_n: float
) -> jax.Array:
    return observe(x, key, sigma_n, stats.asinh_scale)[:, None, :, :]


def validation_observations(
    x: jax.Array, idx: jax.Array, seed: int, stats: Normalization, sigma_n: float
) -> jax.Array:
    keys = jax.vmap(lambda index: jax.random.fold_in(jax.random.PRNGKey(seed), index))(idx)
    z = jax.vmap(lambda image, key: observe(image, key, sigma_n, stats.asinh_scale))(x, keys)
    return z[:, None, :, :]
