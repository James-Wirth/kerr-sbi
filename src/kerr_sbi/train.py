import importlib.metadata
import json
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from kerr_sbi.config import Config, project_path
from kerr_sbi.data import (
    Normalization,
    batch_order,
    fit_normalization,
    load_split,
    prior_log_prob,
    theta_to_u,
    training_observations,
    training_rows,
    u_to_theta,
    validation_observations,
)
from kerr_sbi.dummy import dummy_config
from kerr_sbi.model import Posterior, masked_nll
from kerr_sbi.persistence import exclusive_lock, file_sha256, write_json


class TrainState(eqx.Module):
    model: Posterior
    best_model: Posterior
    opt_state: Any
    noise_key: jax.Array
    shuffle_key: jax.Array
    step: jax.Array
    best_validation: jax.Array
    stale_epochs: jax.Array


def effective_config(cfg: Config, *, dummy: bool, overfit: bool) -> Config:
    if overfit and not dummy:
        raise ValueError("the noiseless overfit experiment requires explicit dummy mode")
    cfg = dummy_config(cfg) if dummy else deepcopy(cfg)
    if dummy:
        for name in ("batch_size", "warmup_steps", "max_steps"):
            cfg["training"][name] = cfg["dummy_training"][name]
    if overfit:
        cfg["training"]["max_steps"] = cfg["dummy_overfit"]["max_steps"]
    return cfg


def optimizer_for(cfg: Config, total_steps: int) -> optax.GradientTransformation:
    settings = cfg["training"]
    warmup = settings["warmup_steps"]
    if not 0 < warmup < total_steps:
        raise ValueError("total steps must exceed the positive warmup length")
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, settings["learning_rate"], warmup, total_steps
    )
    return optax.chain(
        optax.clip_by_global_norm(settings["gradient_clip"]),
        optax.adamw(schedule, weight_decay=settings["weight_decay"]),
    )


def initialize_state(cfg: Config, optimizer: optax.GradientTransformation) -> TrainState:
    model_key, noise_key, shuffle_key = jax.random.split(
        jax.random.PRNGKey(cfg["training"]["seed"]), 3
    )
    model = Posterior(cfg, model_key)
    return TrainState(
        model=model,
        best_model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_inexact_array)),
        noise_key=noise_key,
        shuffle_key=shuffle_key,
        step=jnp.array(0),
        best_validation=jnp.array(jnp.inf),
        stale_epochs=jnp.array(0),
    )


def make_update(
    optimizer: optax.GradientTransformation, stats: Normalization, sigma_n: float
) -> Any:
    @eqx.filter_jit
    def update(
        model: Posterior,
        opt_state: Any,
        key: jax.Array,
        x: jax.Array,
        u: jax.Array,
        mask: jax.Array,
    ) -> tuple:
        next_key, observation_key = jax.random.split(key)
        z = training_observations(x, observation_key, stats, sigma_n)
        loss, gradients = eqx.filter_value_and_grad(masked_nll)(model, z, u, mask)
        updates, opt_state = optimizer.update(
            gradients, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return (
            eqx.apply_updates(model, updates),
            opt_state,
            next_key,
            loss,
            optax.tree.norm(gradients),
        )

    return update


evaluate_batch = eqx.filter_jit(masked_nll)


def evaluate(model: Posterior, z: jax.Array, u: jax.Array, batch_size: int) -> float:
    order, masks = batch_order(len(u), batch_size, jax.random.PRNGKey(0))
    total = 0.0
    for rows, mask in zip(order, masks, strict=True):
        total += float(evaluate_batch(model, z[rows], u[rows], mask)) * int(mask.sum())
    return total / len(u)


def save_checkpoint(
    directory: Path, state: TrainState, identity: dict, history: list, summary: dict
) -> None:
    name = f"checkpoint_{int(state.step):08d}"
    target = directory / name
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=directory) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        eqx.tree_serialise_leaves(stage / "state.eqx", state)
        eqx.tree_serialise_leaves(stage / "best.eqx", state.best_model)
        metadata = {
            "identity": identity,
            "history": history,
            "summary": summary,
            "files": {name: file_sha256(stage / name) for name in ("state.eqx", "best.eqx")},
        }
        write_json(stage / "metadata.json", metadata)
        stage.rename(target)
    write_json(
        directory / "latest.json",
        {"directory": name, "metadata_sha256": file_sha256(target / "metadata.json")},
    )


def load_checkpoint(
    directory: Path, template: TrainState, identity: dict
) -> tuple[TrainState, dict]:
    pointer = json.loads((directory / "latest.json").read_text())
    name = pointer["directory"]
    if Path(name).name != name or not name.startswith("checkpoint_"):
        raise ValueError("invalid checkpoint directory")
    root = directory / name
    if file_sha256(root / "metadata.json") != pointer["metadata_sha256"]:
        raise ValueError("checkpoint metadata hash differs")
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["identity"] != identity:
        raise ValueError("checkpoint config, data, normalization, code or environment differs")
    if set(metadata["files"]) != {"state.eqx", "best.eqx"}:
        raise ValueError("checkpoint file manifest is incomplete")
    for name, digest in metadata["files"].items():
        if name not in ("state.eqx", "best.eqx") or file_sha256(root / name) != digest:
            raise ValueError("checkpoint array hash differs")
    state = eqx.tree_deserialise_leaves(root / "state.eqx", template)
    return state, metadata


def run_training(
    cfg: Config,
    run_name: str,
    *,
    dummy: bool = False,
    overfit: bool = False,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not run_name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in run_name):
        raise ValueError("run name must contain only letters, digits, hyphens or underscores")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop-after must be a positive global step")
    cfg = effective_config(cfg, dummy=dummy, overfit=overfit)
    data = load_split(cfg, "train", allow_dummy=dummy)
    rows, validation = training_rows(data, cfg["training"]["validation_fraction"])
    if overfit:
        count = cfg["dummy_overfit"]["subset_size"]
        if not 2 <= count <= len(rows):
            raise ValueError("overfit subset must fit inside training-only rows")
        rows = rows[:count]
    stats = fit_normalization(data, rows, cfg)
    settings = cfg["training"]
    batch_size = settings["batch_size"]
    if (
        type(batch_size) is not int
        or batch_size < 1
        or settings["max_epochs"] < 1
        or settings["max_steps"] < 0
    ):
        raise ValueError("invalid training batch size, epoch or step limit")
    batches_per_epoch = (len(rows) + batch_size - 1) // batch_size
    total_steps = settings["max_epochs"] * batches_per_epoch
    if settings["max_steps"]:
        total_steps = min(total_steps, settings["max_steps"])
    optimizer = optimizer_for(cfg, total_steps)
    sigma_n = cfg["dummy_overfit"]["sigma_n"] if overfit else cfg["observation"]["sigma_n"]
    max_seconds = cfg["dummy_overfit"]["max_seconds"] if overfit else None
    identity = {
        "config": {key: value for key, value in cfg.items() if key != "project_root"},
        "dummy": dummy,
        "overfit": overfit,
        "effective_sigma_n": sigma_n,
        "total_steps": total_steps,
        "normalization": asdict(stats),
        "train_indices": data.idx[rows].tolist(),
        "validation_indices": data.idx[validation].tolist(),
        "dataset": data.metadata,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "equinox", "optax", "flowjax", "numpy")
        },
        "device": str(jax.devices()[0]),
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("data.py", "model.py", "train.py", "obs_model.py")
        },
    }
    identity = json.loads(json.dumps(identity))
    directory = project_path(cfg, "runs") / run_name
    directory.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory.parent / f".{run_name}.lock"):
        if directory.exists() != resume:
            raise ValueError("use a new run name, or --resume for an existing checkpoint")
        template = initialize_state(cfg, optimizer)
        if resume:
            state, checkpoint = load_checkpoint(directory, template, identity)
            history = checkpoint["history"]
            previous_seconds = checkpoint["summary"]["elapsed_seconds_total"]
            if checkpoint["summary"]["stop_reason"] in (
                "complete",
                "early_stopping",
                "overfit_target",
                "time_limit",
            ):
                return {**checkpoint["summary"], "updates_this_invocation": 0}
            if stop_after is not None and stop_after <= int(state.step):
                raise ValueError("stop-after must be beyond the saved step")
        else:
            state, history, previous_seconds = template, [], 0.0
            directory.mkdir()
            write_json(directory / "run.json", identity)
        initial_step = int(state.step)
        x = jnp.asarray(data.x[rows])
        u = theta_to_u(jnp.asarray(data.theta[rows]), stats)
        validation_u = theta_to_u(jnp.asarray(data.theta[validation]), stats)
        train_z = validation_observations(
            x, jnp.asarray(data.idx[rows]), settings["validation_noise_seed"], stats, sigma_n
        )
        validation_z = validation_observations(
            jnp.asarray(data.x[validation]),
            jnp.asarray(data.idx[validation]),
            settings["validation_noise_seed"],
            stats,
            sigma_n,
        )
        prior_train_nll = float(-prior_log_prob(u, stats).mean())
        prior_validation_nll = float(-prior_log_prob(validation_u, stats).mean())

        def measure(current: TrainState) -> tuple[TrainState, dict]:
            train_loss = evaluate(current.model, train_z, u, batch_size)
            validation_loss = evaluate(current.model, validation_z, validation_u, batch_size)
            if not np.isfinite([train_loss, validation_loss]).all():
                raise FloatingPointError("nonfinite evaluation loss")
            improved = validation_loss < float(current.best_validation)
            current = replace(
                current,
                best_model=current.model if improved else current.best_model,
                best_validation=jnp.asarray(validation_loss)
                if improved
                else current.best_validation,
                stale_epochs=jnp.array(0) if improved else current.stale_epochs + 1,
            )
            record = {
                "step": int(current.step),
                "train_nll": train_loss,
                "validation_nll": validation_loss,
                "prior_train_nll": prior_train_nll,
                "prior_validation_nll": prior_validation_nll,
            }
            print(json.dumps(record), flush=True)
            return current, record

        pending_evaluation = (
            history
            and history[-1]["step"] < int(state.step)
            and (int(state.step) % batches_per_epoch == 0 or int(state.step) == total_steps)
        )
        if not history or pending_evaluation:
            state, record = measure(state)
            history.append(record)
        update = make_update(optimizer, stats, sigma_n)
        durations, gradient_norms, batch_losses = [], [], []
        stop_reason = "complete"
        epoch_order = epoch_masks = None
        active_epoch = None
        try:
            while int(state.step) < total_steps:
                step = int(state.step)
                if stop_after is not None and step >= stop_after:
                    stop_reason = "paused"
                    break
                if (
                    max_seconds is not None
                    and previous_seconds + time.perf_counter() - started >= max_seconds
                ):
                    stop_reason = "time_limit"
                    break
                epoch, batch = divmod(step, batches_per_epoch)
                if epoch != active_epoch:
                    epoch_order, epoch_masks = batch_order(
                        len(rows), batch_size, jax.random.fold_in(state.shuffle_key, epoch)
                    )
                    active_epoch = epoch
                indices, mask = epoch_order[batch], epoch_masks[batch]
                before = time.perf_counter()
                model, opt_state, key, loss, gradient_norm = update(
                    state.model, state.opt_state, state.noise_key, x[indices], u[indices], mask
                )
                loss, gradient_norm = float(loss), float(gradient_norm)
                durations.append(time.perf_counter() - before)
                if not np.isfinite([loss, gradient_norm]).all():
                    raise FloatingPointError("nonfinite training loss or gradient")
                state = replace(
                    state, model=model, opt_state=opt_state, noise_key=key, step=state.step + 1
                )
                batch_losses.append(loss)
                gradient_norms.append(gradient_norm)
                check_target = (
                    overfit and int(state.step) == cfg["dummy_overfit"]["target_check_steps"]
                )
                if (
                    int(state.step) % batches_per_epoch == 0
                    or int(state.step) == total_steps
                    or check_target
                ):
                    state, record = measure(state)
                    history.append(record)
                    if (
                        overfit
                        and int(state.step) >= cfg["dummy_overfit"]["target_check_steps"]
                        and record["train_nll"] < prior_train_nll
                    ):
                        stop_reason = "overfit_target"
                        break
                    if int(state.stale_epochs) >= settings["patience"]:
                        stop_reason = "early_stopping"
                        break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        summary = {
            "run": run_name,
            "dummy": dummy,
            "overfit": overfit,
            "stop_reason": stop_reason,
            "step": int(state.step),
            "updates_this_invocation": int(state.step) - initial_step,
            "elapsed_seconds_total": previous_seconds + time.perf_counter() - started,
            "first_update_seconds_including_compile": durations[0] if durations else None,
            "median_later_update_seconds": float(np.median(durations[1:]))
            if len(durations) > 1
            else None,
            "max_gradient_norm": max(gradient_norms) if gradient_norms else None,
            "last_batch_nll": batch_losses[-1] if batch_losses else None,
            "best_validation_nll": float(state.best_validation),
            "last_evaluation": history[-1],
            "checkpoint_root": str(directory),
        }
        save_checkpoint(directory, state, identity, history, summary)
        write_json(directory / "summary.json", summary)
        return summary


def posterior_check(cfg: Config, run_name: str) -> dict[str, Any]:
    cfg = dummy_config(cfg)
    directory = project_path(cfg, "runs") / run_name
    identity = json.loads((directory / "run.json").read_text())
    if not identity["dummy"] or identity["overfit"]:
        raise ValueError("posterior check requires a noisy dummy training run")
    run_cfg = {**identity["config"], "project_root": cfg["project_root"]}
    state, _ = load_checkpoint(
        directory,
        initialize_state(run_cfg, optimizer_for(run_cfg, identity["total_steps"])),
        identity,
    )
    stats = Normalization(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in identity["normalization"].items()
        }
    )
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
    u = sample(state.best_model)
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
