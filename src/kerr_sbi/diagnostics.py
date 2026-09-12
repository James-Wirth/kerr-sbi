import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kerr_sbi.config import Config, project_path
from kerr_sbi.data import (
    load_split,
    prior_log_prob,
    theta_to_u,
    u_to_theta,
    validation_observations,
)
from kerr_sbi.inference import load_posterior
from kerr_sbi.persistence import exclusive_lock, file_sha256, write_json


def prior_mean(cfg: Config) -> np.ndarray:
    prior = cfg["prior"]
    lower, upper = np.cos(np.deg2rad([prior["incl_max_deg"], prior["incl_min_deg"]]))

    def integral(c: float) -> float:
        return float(c * np.arccos(c) - np.sqrt(1 - c * c))

    inclination = np.rad2deg((integral(upper) - integral(lower)) / (upper - lower))
    return np.array([(prior["a_min"] + prior["a_max"]) / 2, inclination])


def posterior_metrics(theta: np.ndarray, truth: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    if theta.ndim != 3 or theta.shape[1:] != truth.shape or truth.shape[1:] != (2,):
        raise ValueError("posterior draws must have shape (samples, observations, 2)")
    if not np.isfinite(theta).all() or not np.isfinite(truth).all():
        raise FloatingPointError("nonfinite posterior draws or truth")
    mean = theta.mean(axis=0)
    coverage = {}
    for level in (0.68, 0.9, 0.95):
        low, high = np.quantile(theta, [(1 - level) / 2, (1 + level) / 2], axis=0)
        coverage[str(level)] = np.mean((truth >= low) & (truth <= high), axis=0).tolist()
    return {
        "rmse": np.sqrt(np.mean((mean - truth) ** 2, axis=0)).tolist(),
        "prior_rmse": np.sqrt(np.mean((baseline - truth) ** 2, axis=0)).tolist(),
        "mean_posterior_std": theta.std(axis=0).mean(axis=0).tolist(),
        "coverage": coverage,
    }


def rank_band(observations: int, bins: int) -> tuple[int, int]:
    p = 1 / bins
    k = np.arange(observations + 1)
    log_pmf = np.array(
        [
            math.lgamma(observations + 1)
            - math.lgamma(int(value) + 1)
            - math.lgamma(observations - int(value) + 1)
            + value * math.log(p)
            + (observations - value) * math.log1p(-p)
            for value in k
        ]
    )
    cdf = np.cumsum(np.exp(log_pmf))
    return int(np.searchsorted(cdf, 0.005)), int(np.searchsorted(cdf, 0.995))


def save_figures(
    directory: Path,
    theta: np.ndarray,
    truth: np.ndarray,
    ranks: np.ndarray,
    settings: dict[str, Any],
) -> None:
    mean, std = theta.mean(axis=0), theta.std(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for dimension, (axis, label) in enumerate(
        zip(axes, ("Spin a", "Inclination (degrees)"), strict=True)
    ):
        axis.errorbar(
            truth[:, dimension],
            mean[:, dimension],
            yerr=std[:, dimension],
            fmt="none",
            alpha=0.22,
            color="#155ddb",
        )
        dots = axis.scatter(
            truth[:, dimension], mean[:, dimension], c=truth[:, 1 - dimension], s=15, cmap="viridis"
        )
        lower, upper = truth[:, dimension].min(), truth[:, dimension].max()
        axis.plot([lower, upper], [lower, upper], color="#666666", linestyle="--", linewidth=1)
        axis.set(xlabel=f"True {label}", ylabel="Posterior mean ± one standard deviation")
        fig.colorbar(dots, ax=axis, label="Inclination (degrees)" if dimension == 0 else "Spin a")
    fig.suptitle("Development observations · posterior means and uncertainty")
    fig.savefig(directory / "posterior_means.png", dpi=settings["figure_dpi"])
    plt.close(fig)
    bins, draws = settings["rank_bins"], settings["posterior_samples"]
    low, high = rank_band(len(truth), bins)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for dimension, (axis, label) in enumerate(
        zip(axes, ("Spin", "Cosine of inclination"), strict=True)
    ):
        axis.axhspan(low, high, color="#007d74", alpha=0.12, label="Pointwise 99% binomial band")
        axis.axhline(len(truth) / bins, color="#007d74", linestyle="--", linewidth=1)
        axis.hist(
            ranks[:, dimension],
            bins=np.linspace(-0.5, draws + 0.5, bins + 1),
            color="#155ddb",
            alpha=0.8,
        )
        axis.set(xlabel=f"{label} posterior rank (u-space)", ylabel="Observations")
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory development ranks · final test set remains reserved")
    fig.savefig(directory / "ranks.png", dpi=settings["figure_dpi"])
    plt.close(fig)


def development_check(cfg: Config, run_name: str) -> dict[str, Any]:
    if not run_name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in run_name):
        raise ValueError("invalid run name")
    directory = project_path(cfg, "runs") / run_name
    with exclusive_lock(directory.parent / f".{run_name}.lock"):
        posterior = load_posterior(directory)
        identity = posterior.identity
        if identity["dummy"] or identity["overfit"]:
            raise ValueError("development checks require a noisy physical run")
        settings = cfg["development_diagnostics"]
        count, batch_size = settings["posterior_samples"], settings["batch_size"]
        if not (type(count) is int and count >= 19 and type(batch_size) is int and batch_size > 0):
            raise ValueError("invalid diagnostic sample or batch count")
        if settings["rank_bins"] < 2 or (count + 1) % settings["rank_bins"]:
            raise ValueError("rank bins must evenly divide the number of possible ranks")
        recorded = {**identity["config"], "project_root": cfg["project_root"]}
        dev = load_split(recorded, "dev")
        if dev.metadata["seed"] == identity["dataset"]["seed"]:
            raise ValueError("development and training prior seeds overlap")
        if len(dev.idx) != dev.metadata["n_requested"] or len(dev.idx) < 2:
            raise ValueError("development checks require a complete held-out split")
        provenance = {
            "checkpoint": json.loads((directory / "latest.json").read_text()),
            "dataset": dev.metadata,
            "settings": settings,
            "source_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("diagnostics.py", "inference.py", "model.py", "data.py")
            },
        }
        destination = directory / "diagnostics"
        if destination.exists():
            report = json.loads((destination / "summary.json").read_text())
            if report.get("provenance") != provenance:
                raise ValueError(
                    "existing diagnostics use a different checkpoint, dataset or implementation"
                )
            return report
        stats = posterior.normalization
        z = validation_observations(
            jnp.asarray(dev.x),
            jnp.asarray(dev.idx),
            settings["noise_seed"],
            stats,
            identity["effective_sigma_n"],
        )
        truth_u = theta_to_u(jnp.asarray(dev.theta), stats)
        permutation = np.random.default_rng(settings["shuffle_seed"]).permutation(len(z))
        permutation = np.roll(permutation, 1)[np.argsort(permutation)]
        log_prob = eqx.filter_jit(lambda model, images, values: model.log_prob(images, values))
        sample = eqx.filter_jit(lambda model, images, key: model.sample(images, key, count))
        matched, shuffled, draws = [], [], []
        for start in range(0, len(z), batch_size):
            stop = min(start + batch_size, len(z))
            matched.append(
                np.asarray(log_prob(posterior.model, z[start:stop], truth_u[start:stop]))
            )
            shuffled.append(
                np.asarray(
                    log_prob(posterior.model, z[permutation[start:stop]], truth_u[start:stop])
                )
            )
            key = jax.random.fold_in(jax.random.key(settings["posterior_seed"]), start)
            draws.append(np.asarray(sample(posterior.model, z[start:stop], key)))
            print(f"Development diagnostics: {stop}/{len(z)} observations", flush=True)
        u = np.concatenate(draws, axis=1)
        theta = np.asarray(u_to_theta(jnp.asarray(u), stats))
        ranks = np.sum(u < np.asarray(truth_u)[None], axis=0)
        nll, shuffled_nll = -np.concatenate(matched), -np.concatenate(shuffled)
        prior_nll = -np.asarray(prior_log_prob(truth_u, stats))
        if not np.isfinite(u).all() or not np.isfinite([nll, shuffled_nll, prior_nll]).all():
            raise FloatingPointError("nonfinite development density or samples")
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "scientific_calibration": False,
            "split": "dev",
            "n_observations": len(dev.idx),
            "posterior_samples": count,
            "nll": float(nll.mean()),
            "prior_nll": float(prior_nll.mean()),
            "shuffled_nll": float(shuffled_nll.mean()),
            "prior_gain_standard_error": float(np.std(prior_nll - nll, ddof=1) / np.sqrt(len(nll))),
            "conditioning_gain_standard_error": float(
                np.std(shuffled_nll - nll, ddof=1) / np.sqrt(len(nll))
            ),
            **posterior_metrics(theta, dev.theta, prior_mean(recorded)),
            "provenance": provenance,
        }
        with tempfile.TemporaryDirectory(prefix=".diagnostics-", dir=directory) as temporary:
            stage = Path(temporary) / "diagnostics"
            stage.mkdir()
            np.savez(
                stage / "samples.npz",
                u=u,
                theta=theta,
                ranks=ranks,
                truth_theta=dev.theta,
                idx=dev.idx,
                nll=nll,
                shuffled_nll=shuffled_nll,
                prior_nll=prior_nll,
                shuffled_indices=permutation,
            )
            save_figures(stage, theta, dev.theta, ranks, settings)
            write_json(stage / "summary.json", report)
            stage.rename(destination)
        return report
