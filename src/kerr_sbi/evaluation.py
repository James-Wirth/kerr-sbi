import csv
import json
import time
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from kerr_sbi.checkpoint import read_checkpoint
from kerr_sbi.data import (
    load_split,
    prior_log_prob,
    theta_to_u,
    u_to_theta,
    validation_observations,
)
from kerr_sbi.inference import load_posterior
from kerr_sbi.obs_model import add_noise, network_transform
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.reference import compare_samples, mass_summary

WILSON_Z95 = 1.959963984540054
EVALUATION_SPLIT = "dev"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = tomllib.loads(path.read_text())
    development, reference = protocol["development"], protocol["reference"]
    if development["split"] != EVALUATION_SPLIT:
        raise ValueError(
            "only the development split can be evaluated; the reserved test split "
            "requires a separately frozen final protocol"
        )
    seeds = protocol["sampling"]["posterior_seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("posterior seeds must be unique and nonempty")
    for value in (development["count"], development["draws"], development["batch_size"]):
        if type(value) is not int or value < 1:
            raise ValueError("development counts must be positive integers")
    if type(reference["draws"]) is not int or reference["draws"] < 1:
        raise ValueError("reference draws must be a positive integer")
    levels = protocol["analysis"]["coverage_levels"]
    if not levels or not all(0 < level < 1 for level in levels):
        raise ValueError("coverage levels must lie strictly between 0 and 1")
    return protocol


def wilson(successes: Any, n: int, z: float = WILSON_Z95) -> np.ndarray:
    successes = np.asarray(successes, dtype=float)
    if n < 1 or np.any((successes < 0) | (successes > n)):
        raise ValueError("Invalid binomial count")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    delta = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return np.stack([center - delta, center + delta], axis=-1)


def interval_statistics(
    theta: np.ndarray,
    truth: np.ndarray,
    nll: np.ndarray,
    prior_nll: np.ndarray,
    levels: Sequence[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if theta.shape[1:] != truth.shape or not np.isfinite(theta).all():
        raise ValueError("Invalid draws")
    mean = theta.mean(axis=0, dtype=np.float64)
    std = theta.std(axis=0, dtype=np.float64)
    low, high = np.quantile(theta, [0.05, 0.95], axis=0)
    per = {"squared_error": (mean - truth) ** 2, "std": std, "width90": high - low, "nll": nll}
    report = {
        "n": len(truth),
        "nll": float(nll.mean()),
        "prior_nll": float(prior_nll.mean()),
        "nll_gain": float((prior_nll - nll).mean()),
        "nll_gain_se": float((prior_nll - nll).std(ddof=1) / np.sqrt(len(truth))),
        "rmse": np.sqrt(per["squared_error"].mean(0)).tolist(),
        "mean_std": std.mean(0).tolist(),
        "mean_width90": (high - low).mean(0).tolist(),
        "mean_bias": (mean - truth).mean(0).tolist(),
        "coverage": {},
    }
    for level in levels:
        lo, hi = np.quantile(theta, [(1 - level) / 2, (1 + level) / 2], axis=0)
        hit = (truth >= lo) & (truth <= hi)
        per[f"coverage_{level}"] = hit.astype(float)
        report["coverage"][str(level)] = {
            "value": hit.mean(0).tolist(),
            "count": hit.sum(0).tolist(),
            "wilson95": wilson(hit.sum(0), len(truth)).tolist(),
        }
    return report, per


def paired_bootstrap(
    current: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    indices: np.ndarray,
    levels: Sequence[float],
) -> dict[str, Any]:
    result = {}
    for key in ("nll", "squared_error", "width90", *(f"coverage_{level}" for level in levels)):
        x, y = current[key], baseline[key]
        if key == "squared_error":
            delta = np.sqrt(x.mean(0)) - np.sqrt(y.mean(0))
            draws = np.sqrt(x[indices].mean(1)) - np.sqrt(y[indices].mean(1))
            label = "rmse"
        else:
            delta = (x - y).mean(0)
            draws = (x - y)[indices].mean(1)
            label = key
        result[label] = {
            "difference": np.asarray(delta).tolist(),
            "paired_bootstrap95": np.moveaxis(
                np.quantile(draws, [0.025, 0.975], axis=0), 0, -1
            ).tolist(),
        }
    return result


def bin_mask(values: np.ndarray, lower: float, upper: float, last: bool) -> np.ndarray:
    return (values >= lower) & ((values <= upper) if last else (values < upper))


def bin_rows(
    run: str,
    theta: np.ndarray,
    truth: np.ndarray,
    nll: np.ndarray,
    prior_nll: np.ndarray,
    analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    levels = analysis["coverage_levels"]
    headline = str(analysis["headline_coverage_level"])
    for dimension, name, edges in (
        (1, "inclination", analysis["inclination_bin_edges_deg"]),
        (0, "spin", analysis["spin_bin_edges"]),
    ):
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            mask = bin_mask(truth[:, dimension], lower, upper, upper == edges[-1])
            if mask.sum() < 2:
                raise ValueError(f"{name} bin [{lower}, {upper}] has fewer than two observations")
            report, _ = interval_statistics(
                theta[:, mask], truth[mask], nll[mask], prior_nll[mask], levels
            )
            coverage = report["coverage"][headline]
            rows.append(
                {
                    "run": run,
                    "bin_variable": name,
                    "lower": lower,
                    "upper": upper,
                    "n": int(mask.sum()),
                    "nll": report["nll"],
                    "spin_rmse": report["rmse"][0],
                    "inclination_rmse_deg": report["rmse"][1],
                    "coverage_level": float(headline),
                    "spin_coverage": coverage["value"][0],
                    "spin_coverage_ci_low": coverage["wilson95"][0][0],
                    "spin_coverage_ci_high": coverage["wilson95"][0][1],
                    "inclination_coverage": coverage["value"][1],
                    "inclination_coverage_ci_low": coverage["wilson95"][1][0],
                    "inclination_coverage_ci_high": coverage["wilson95"][1][1],
                    "spin_width90": report["mean_width90"][0],
                    "inclination_width90_deg": report["mean_width90"][1],
                }
            )
    return rows


def check_prior_support(theta: np.ndarray, prior: Mapping[str, float]) -> None:
    if not np.isfinite(theta).all():
        raise ValueError("Nonfinite posterior draws")
    if (
        theta[..., 0].min() < prior["a_min"]
        or theta[..., 0].max() > prior["a_max"] + 1e-6
        or theta[..., 1].min() < prior["incl_min_deg"] - 1e-4
        or theta[..., 1].max() > prior["incl_max_deg"] + 1e-4
    ):
        raise ValueError("Draws outside physical prior")


def reference_observations(
    dev: Any, cases: Sequence[Mapping[str, Any]], sigma_n: float
) -> np.ndarray:
    rows = []
    for case in cases:
        row = int(np.flatnonzero(dev.idx == case["dev_idx"])[0])
        if not np.array_equal(dev.theta[row], case["truth"]):
            raise ValueError("Reference truth mismatch")
        key = jax.random.fold_in(jax.random.PRNGKey(case["noise_seed"]), int(dev.idx[row]))
        rows.append(np.asarray(add_noise(jnp.asarray(dev.x[row]), key, sigma_n)))
    return np.stack(rows)


def evaluate_run(
    root: Path, run_name: str, protocol: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    started = time.monotonic()
    development, reference = protocol["development"], protocol["reference"]
    run = root / "runs" / run_name
    posterior = load_posterior(run)
    identity = posterior.identity
    if identity["dummy"] or identity["overfit"]:
        raise ValueError("evaluation requires a noisy physical training run")
    checkpoint, metadata = read_checkpoint(run, identity)
    if metadata["summary"]["stop_reason"] not in ("complete", "early_stopping"):
        raise ValueError("Training has not completed")
    cfg = {**identity["config"], "project_root": root}
    dev = load_split(cfg, development["split"])
    if dev.metadata["seed"] == identity["dataset"]["seed"]:
        raise ValueError("development and training prior seeds overlap")
    if len(dev.idx) != development["count"]:
        raise ValueError("Wrong development set")
    stats, sigma_n = posterior.normalization, identity["effective_sigma_n"]
    z = validation_observations(
        jnp.asarray(dev.x),
        jnp.asarray(dev.idx),
        development["observation_noise_seed"],
        stats,
        sigma_n,
    )
    truth_u = theta_to_u(jnp.asarray(dev.theta), stats)
    reference_dir = root / reference["path"]
    report = json.loads((reference_dir / "summary.json").read_text())
    if not report["reference_certified"]:
        raise ValueError("Reference has not passed its finite numerical checks")
    with np.load(reference_dir / "posteriors.npz") as data:
        a, c, mass, observations = [data[k] for k in ("a", "c", "reference_mass", "observations")]
    cases = report["cases"]
    if len(cases) != reference["cases"] or not np.allclose(mass.sum((1, 2)), 1):
        raise ValueError("Invalid reference masses")
    if not np.array_equal(observations, reference_observations(dev, cases, sigma_n)):
        raise ValueError("Reference observations differ from their recorded data/noise identities")
    reference_z = network_transform(jnp.asarray(observations), stats.asinh_scale)[:, None]
    batch = development["batch_size"]
    log_prob = eqx.filter_jit(lambda model, images, values: model.log_prob(images, values))
    dev_sample = eqx.filter_jit(
        lambda model, images, key: model.sample(images, key, development["draws"])
    )
    reference_sample = eqx.filter_jit(
        lambda model, images, key: model.sample(images, key, reference["draws"])
    )
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run": run_name,
        "protocol": protocol,
        "evaluator_sha256": file_sha256(Path(__file__)),
        "checkpoint": json.loads((run / "latest.json").read_text()),
        "best_weights_sha256": file_sha256(checkpoint / "best.eqx"),
        "training_source_sha256": identity["source_sha256"],
        "dataset_arrays": dev.metadata["arrays"],
        "reference_archive_sha256": file_sha256(reference_dir / "posteriors.npz"),
    }
    provenance_path = destination / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Existing evaluation has different provenance")
    write_json(provenance_path, provenance)
    nll = np.concatenate(
        [
            np.asarray(
                -log_prob(posterior.model, z[start : start + batch], truth_u[start : start + batch])
            )
            for start in range(0, len(z), batch)
        ]
    )
    prior_nll = np.asarray(-prior_log_prob(truth_u, stats))
    if not np.isfinite([nll, prior_nll]).all():
        raise ValueError("Nonfinite density")
    reference_summary = mass_summary(a, c, mass)
    for seed in protocol["sampling"]["posterior_seeds"]:
        artifact = destination / f"samples_{seed}.npz"
        summary_path = destination / f"reference_{seed}.json"
        if artifact.exists() and summary_path.exists():
            if json.loads(summary_path.read_text())["samples_sha256"] != file_sha256(artifact):
                raise ValueError("Evaluation cache hash mismatch")
            continue
        draws = []
        for start in range(0, len(z), batch):
            key = jax.random.fold_in(jax.random.key(seed), start)
            u = dev_sample(posterior.model, z[start : start + batch], key)
            draws.append(np.asarray(u_to_theta(u, stats)))
        theta = np.concatenate(draws, axis=1)
        reference_theta = np.asarray(
            u_to_theta(reference_sample(posterior.model, reference_z, jax.random.key(seed)), stats)
        )
        for values in (theta, reference_theta):
            check_prior_support(values, cfg["prior"])
        np.savez(
            artifact,
            theta=theta,
            reference_theta=reference_theta,
            nll=nll,
            prior_nll=prior_nll,
            truth_theta=dev.theta,
            idx=dev.idx,
        )
        comparison = compare_samples(a, c, mass, reference_theta)
        reference_std = np.asarray(reference_summary["std"])
        comparison["signed_mean_shift_in_std"] = (
            (reference_theta.mean(0) - reference_summary["mean"]) / reference_std
        ).tolist()
        comparison["std_ratio"] = (reference_theta.std(0) / reference_std).tolist()
        write_json(
            summary_path,
            {
                "sampling_seed": seed,
                "cases": cases,
                "comparison": comparison,
                "reference_summary": reference_summary,
                "samples_sha256": file_sha256(artifact),
            },
        )
    completion = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "provenance_sha256": file_sha256(provenance_path),
    }
    write_json(destination / "complete.json", completion)
    return completion


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory: Path, runs: Sequence[str], protocol: Mapping[str, Any]) -> dict[str, Any]:
    seeds = protocol["sampling"]["posterior_seeds"]
    analysis = protocol["analysis"]
    levels = analysis["coverage_levels"]
    headline = str(analysis["headline_coverage_level"])
    models, per_run, bins, reference_rows, table, inputs = {}, {}, [], [], [], {}
    comparisons: dict[str, dict[int, dict[str, Any]]] = {}
    for name in runs:
        destination = directory / name
        if not (destination / "complete.json").exists():
            raise ValueError(f"Evaluation incomplete: {name}")
        models[name] = {"repetitions": {}, "reference": {}}
        per_run[name], comparisons[name] = {}, {}
        for seed in seeds:
            path, reference_path = (
                destination / f"samples_{seed}.npz",
                destination / (f"reference_{seed}.json"),
            )
            reference = json.loads(reference_path.read_text())
            if file_sha256(path) != reference["samples_sha256"]:
                raise ValueError("Samples changed")
            inputs[f"{name}/{path.name}"] = reference["samples_sha256"]
            with np.load(path) as data:
                theta, truth, nll, prior_nll = [
                    data[k] for k in ("theta", "truth_theta", "nll", "prior_nll")
                ]
            report, per = interval_statistics(theta, truth, nll, prior_nll, levels)
            models[name]["repetitions"][str(seed)] = report
            per_run[name][seed] = per
            if seed == seeds[0]:
                bins.extend(bin_rows(name, theta, truth, nll, prior_nll, analysis))
            comparison = reference["comparison"]
            comparisons[name][seed] = comparison
            models[name]["reference"][str(seed)] = {
                key: np.median(comparison[key], axis=0).tolist()
                for key in (
                    "marginal_cdf_change",
                    "mean_shift_in_std",
                    "relative_std_change",
                    "signed_mean_shift_in_std",
                    "std_ratio",
                )
            }
            for index, case in enumerate(reference["cases"]):
                reference_rows.append(
                    {
                        "run": name,
                        "sampling_seed": seed,
                        "case": index,
                        "dev_idx": case["dev_idx"],
                        "observation_noise_seed": case["noise_seed"],
                        "true_spin": case["truth"][0],
                        "true_inclination_deg": case["truth"][1],
                        "spin_cdf_error": comparison["marginal_cdf_change"][index][0],
                        "cos_inclination_cdf_error": comparison["marginal_cdf_change"][index][1],
                        "spin_signed_mean_shift_ref_std": comparison["signed_mean_shift_in_std"][
                            index
                        ][0],
                        "inclination_signed_mean_shift_ref_std": comparison[
                            "signed_mean_shift_in_std"
                        ][index][1],
                        "spin_std_ratio": comparison["std_ratio"][index][0],
                        "inclination_std_ratio": comparison["std_ratio"][index][1],
                    }
                )
        first = models[name]["repetitions"][str(seeds[0])]
        first_reference = models[name]["reference"][str(seeds[0])]
        if len(seeds) > 1:
            second = models[name]["repetitions"][str(seeds[1])]
            second_reference = models[name]["reference"][str(seeds[1])]
            change = {
                key: np.abs(np.array(second[key]) - first[key]).tolist()
                for key in ("rmse", "mean_std", "mean_width90")
            }
            change["coverage"] = np.abs(
                np.array(second["coverage"][headline]["value"])
                - first["coverage"][headline]["value"]
            ).tolist()
            change["median_reference_cdf"] = np.abs(
                np.array(second_reference["marginal_cdf_change"])
                - first_reference["marginal_cdf_change"]
            ).tolist()
            models[name]["mc_absolute_repeat_change"] = change
        table.append(
            {
                "run": name,
                "nll": first["nll"],
                "prior_nll": first["prior_nll"],
                "nll_gain": first["nll_gain"],
                "spin_rmse": first["rmse"][0],
                "inclination_rmse_deg": first["rmse"][1],
                "coverage_level": float(headline),
                "spin_coverage": first["coverage"][headline]["value"][0],
                "inclination_coverage": first["coverage"][headline]["value"][1],
                "spin_width90": first["mean_width90"][0],
                "inclination_width90_deg": first["mean_width90"][1],
                "median_spin_cdf_error": first_reference["marginal_cdf_change"][0],
                "median_cos_inclination_cdf_error": first_reference["marginal_cdf_change"][1],
                "median_spin_std_ratio": first_reference["std_ratio"][0],
                "median_inclination_std_ratio": first_reference["std_ratio"][1],
            }
        )
    output: dict[str, Any] = {
        "protocol": protocol,
        "runs": list(runs),
        "baseline": runs[0],
        "analysis_source_sha256": file_sha256(Path(__file__)),
        "models": models,
        "input_sha256": inputs,
    }
    primary = seeds[0]
    count = len(per_run[runs[0]][primary]["nll"])
    if len(runs) > 1:
        rng = np.random.default_rng(analysis["bootstrap_seed"])
        indices = rng.integers(0, count, (analysis["bootstrap_repeats"], count))
        output["paired_minus_baseline"] = {
            name: paired_bootstrap(
                per_run[name][primary], per_run[runs[0]][primary], indices, levels
            )
            for name in runs[1:]
        }
        output["run_sd"] = {
            key: np.std(
                [models[name]["repetitions"][str(primary)][key] for name in runs], axis=0, ddof=1
            ).tolist()
            for key in ("nll", "rmse", "mean_std", "mean_width90")
        }
    cdf = np.array([comparisons[name][primary]["marginal_cdf_change"] for name in runs])
    signed = np.array([comparisons[name][primary]["signed_mean_shift_in_std"] for name in runs])
    ratios = np.array([comparisons[name][primary]["std_ratio"] for name in runs])
    threshold = analysis["descriptive_cdf_threshold"]
    repeatability = {
        "per_case_cdf_min_across_runs": cdf.min(0).tolist(),
        "per_case_cdf_max_across_runs": cdf.max(0).tolist(),
        "median_casewise_run_range": np.median(np.ptp(cdf, axis=0), axis=0).tolist(),
        "same_mean_error_sign_all_runs_count": np.sum(
            np.all(signed > 0, axis=0) | np.all(signed < 0, axis=0), axis=0
        ).tolist(),
        "wider_than_reference_all_runs_count": np.sum(np.all(ratios > 1, axis=0), axis=0).tolist(),
        "cdf_error_above_threshold_all_runs_count": np.sum(
            np.all(cdf > threshold, axis=0), axis=0
        ).tolist(),
        "descriptive_cdf_threshold": threshold,
        "scope": (
            "Descriptive agreement with a fixed development benchmark, not an acceptance "
            "threshold or population failure rate."
        ),
    }
    if len(seeds) > 1:
        repeat = np.array([comparisons[name][seeds[1]]["marginal_cdf_change"] for name in runs])
        repeatability["median_casewise_mc_repeat_change"] = np.median(
            np.abs(cdf - repeat), axis=(0, 1)
        ).tolist()
        repeatability["max_casewise_mc_repeat_change"] = np.max(
            np.abs(cdf - repeat), axis=(0, 1)
        ).tolist()
    output["reference_repeatability"] = repeatability
    write_json(directory / "summary.json", output)
    write_csv(directory / "comparison.csv", table)
    write_csv(directory / "bins.csv", bins)
    write_csv(directory / "reference_cases.csv", reference_rows)
    return output
