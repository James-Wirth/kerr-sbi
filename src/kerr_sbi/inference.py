import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from kerr_sbi.checkpoint import read_checkpoint
from kerr_sbi.config import Config, project_path
from kerr_sbi.data import (
    Normalization,
    load_split,
    theta_to_u,
    u_to_theta,
    validation_observations,
)
from kerr_sbi.dummy import dummy_config
from kerr_sbi.model import Posterior
from kerr_sbi.persistence import write_json


@dataclass(frozen=True)
class LoadedPosterior:
    model: Posterior
    normalization: Normalization
    identity: dict[str, Any]


def load_posterior(directory: Path) -> LoadedPosterior:
    identity = json.loads((directory / "run.json").read_text())
    root, metadata = read_checkpoint(directory, identity)
    cfg = identity["config"]
    model_key, _, _ = jax.random.split(jax.random.PRNGKey(cfg["training"]["seed"]), 3)
    model = eqx.tree_deserialise_leaves(root / "best.eqx", Posterior(cfg, model_key))
    stats = Normalization(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in identity["normalization"].items()
        }
    )
    return LoadedPosterior(model, stats, metadata["identity"])


def posterior_check(cfg: Config, run_name: str) -> dict[str, Any]:
    cfg = dummy_config(cfg)
    directory = project_path(cfg, "runs") / run_name
    identity = json.loads((directory / "run.json").read_text())
    if not identity["dummy"] or identity["overfit"]:
        raise ValueError("posterior check requires a noisy dummy training run")
    posterior = load_posterior(directory)
    stats = posterior.normalization
    test = load_split(cfg, "test", allow_dummy=True)
    if test.metadata["seed"] == identity["dataset"]["seed"]:
        raise ValueError("test and training seeds overlap")
    settings = cfg["dummy_training"]
    z = validation_observations(
        jnp.asarray(test.x),
        jnp.asarray(test.idx),
        settings["posterior_seed"],
        stats,
        identity["effective_sigma_n"],
    )
    sample = eqx.filter_jit(
        lambda model: model.sample(
            z, jax.random.key(settings["posterior_seed"]), settings["posterior_samples"]
        )
    )
    u = sample(posterior.model)
    theta = u_to_theta(u, stats)
    truth = theta_to_u(jnp.asarray(test.theta), stats)
    ranks = jnp.sum(u < truth[None], axis=0)
    if not np.isfinite(theta).all() or not np.isfinite(u).all():
        raise FloatingPointError("nonfinite posterior samples")
    prior = cfg["prior"]
    if not (
        np.all(theta[..., 0] >= prior["a_min"])
        and np.all(theta[..., 0] <= prior["a_max"])
        and np.all(theta[..., 1] >= prior["incl_min_deg"] - 1e-4)
        and np.all(theta[..., 1] <= prior["incl_max_deg"] + 1e-4)
    ):
        raise ValueError("posterior samples outside prior bounds")
    np.savez(
        directory / "dummy_posterior_check.npz",
        u=np.asarray(u),
        theta=np.asarray(theta),
        ranks=np.asarray(ranks),
        truth_theta=test.theta,
        idx=test.idx,
    )
    report = {
        "dummy": True,
        "scientific_calibration": False,
        "samples_shape": list(u.shape),
        "ranks_shape": list(ranks.shape),
        "test_dataset": test.metadata,
        "posterior_seed": settings["posterior_seed"],
    }
    write_json(directory / "dummy_posterior_check.json", report)
    return {key: value for key, value in report.items() if key != "test_dataset"}
