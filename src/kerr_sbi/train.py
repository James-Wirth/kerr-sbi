import importlib.metadata
import json
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from kerr_sbi.checkpoint import EvaluationRecord, TrainingSummary
from kerr_sbi.checkpoint import TrainState as TrainState
from kerr_sbi.checkpoint import load_checkpoint as load_checkpoint
from kerr_sbi.checkpoint import save_checkpoint as save_checkpoint
from kerr_sbi.config import Config, project_path
from kerr_sbi.data import (
    Normalization,
    SplitData,
    batch_order,
    fit_normalization,
    load_split,
    prior_log_prob,
    theta_to_u,
    training_observations,
    training_rows,
    validation_observations,
)
from kerr_sbi.dummy import dummy_config
from kerr_sbi.inference import posterior_check as posterior_check
from kerr_sbi.model import Posterior, masked_nll
from kerr_sbi.persistence import exclusive_lock, file_sha256, write_json
from kerr_sbi.telemetry import EventSink, TrainingEvents


@dataclass(frozen=True)
class PreparedRun:
    cfg: Config
    data: SplitData
    rows: np.ndarray
    validation: np.ndarray
    stats: Normalization
    optimizer: optax.GradientTransformation
    batches_per_epoch: int
    total_steps: int
    sigma_n: float
    max_seconds: float | None
    identity: dict[str, Any]
    directory: Path


@dataclass(frozen=True)
class RunProgress:
    state: TrainState
    history: list[EvaluationRecord]
    previous_seconds: float
    finished_summary: TrainingSummary | None = None


@dataclass(frozen=True)
class EvaluationData:
    x: jax.Array
    u: jax.Array
    train_z: jax.Array
    validation_u: jax.Array
    validation_z: jax.Array
    prior_train_nll: float
    prior_validation_nll: float


UpdateResult = tuple[Posterior, optax.OptState, jax.Array, jax.Array, jax.Array]
Update = Callable[
    [Posterior, optax.OptState, jax.Array, jax.Array, jax.Array, jax.Array], UpdateResult
]


def overfit_settings(cfg: Config, dummy: bool) -> dict[str, Any]:
    return cfg["dummy_overfit" if dummy else "physical_overfit"]


def effective_config(cfg: Config, *, dummy: bool, overfit: bool) -> Config:
    cfg = dummy_config(cfg) if dummy else deepcopy(cfg)
    if dummy:
        for name in ("batch_size", "warmup_steps", "max_steps"):
            cfg["training"][name] = cfg["dummy_training"][name]
    if overfit:
        cfg["training"].update(
            {
                key: value
                for key, value in overfit_settings(cfg, dummy).items()
                if key in cfg["training"]
            }
        )
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
) -> Update:
    @eqx.filter_jit
    def update(
        model: Posterior,
        opt_state: optax.OptState,
        key: jax.Array,
        x: jax.Array,
        u: jax.Array,
        mask: jax.Array,
    ) -> UpdateResult:
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


def prepare_run(
    cfg: Config, run_name: str, *, dummy: bool, overfit: bool, stop_after: int | None
) -> PreparedRun:
    if not run_name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in run_name):
        raise ValueError("run name must contain only letters, digits, hyphens or underscores")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop-after must be a positive global step")
    cfg = effective_config(cfg, dummy=dummy, overfit=overfit)
    data = load_split(cfg, "train", allow_dummy=dummy)
    rows, validation = training_rows(data, cfg["training"]["validation_fraction"])
    if overfit:
        count = overfit_settings(cfg, dummy)["subset_size"]
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
    sigma_n = overfit_settings(cfg, dummy)["sigma_n"] if overfit else cfg["observation"]["sigma_n"]
    max_seconds = overfit_settings(cfg, dummy)["max_seconds"] if overfit else None
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
            for name in (
                "data.py",
                "model.py",
                "train.py",
                "obs_model.py",
                "checkpoint.py",
                "inference.py",
                "persistence.py",
                "telemetry.py",
                "experiments.py",
            )
        },
    }
    identity = json.loads(json.dumps(identity))
    directory = project_path(cfg, "runs") / run_name
    directory.parent.mkdir(parents=True, exist_ok=True)
    return PreparedRun(
        cfg=cfg,
        data=data,
        rows=rows,
        validation=validation,
        stats=stats,
        optimizer=optimizer,
        batches_per_epoch=batches_per_epoch,
        total_steps=total_steps,
        sigma_n=sigma_n,
        max_seconds=max_seconds,
        identity=identity,
        directory=directory,
    )


def create_or_restore_run(run: PreparedRun, *, resume: bool, stop_after: int | None) -> RunProgress:
    cfg, optimizer, identity, directory = run.cfg, run.optimizer, run.identity, run.directory
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
            return RunProgress(
                state,
                history,
                previous_seconds,
                {**checkpoint["summary"], "updates_this_invocation": 0},
            )
        if stop_after is not None and stop_after <= int(state.step):
            raise ValueError("stop-after must be beyond the saved step")
    else:
        state, history, previous_seconds = template, [], 0.0
        directory.mkdir()
        write_json(directory / "run.json", identity)
    return RunProgress(state, history, previous_seconds)


def prepare_evaluation(run: PreparedRun) -> EvaluationData:
    data, rows, validation = run.data, run.rows, run.validation
    stats, sigma_n, settings = run.stats, run.sigma_n, run.cfg["training"]
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
    return EvaluationData(
        x, u, train_z, validation_u, validation_z, prior_train_nll, prior_validation_nll
    )


def measure(
    current: TrainState, data: EvaluationData, batch_size: int
) -> tuple[TrainState, EvaluationRecord]:
    train_loss = evaluate(current.model, data.train_z, data.u, batch_size)
    validation_loss = evaluate(current.model, data.validation_z, data.validation_u, batch_size)
    if not np.isfinite([train_loss, validation_loss]).all():
        raise FloatingPointError("nonfinite evaluation loss")
    improved = validation_loss < float(current.best_validation)
    current = replace(
        current,
        best_model=current.model if improved else current.best_model,
        best_validation=jnp.asarray(validation_loss) if improved else current.best_validation,
        stale_epochs=jnp.array(0) if improved else current.stale_epochs + 1,
    )
    record: EvaluationRecord = {
        "step": int(current.step),
        "train_nll": train_loss,
        "validation_nll": validation_loss,
        "prior_train_nll": data.prior_train_nll,
        "prior_validation_nll": data.prior_validation_nll,
    }
    print(json.dumps(record), flush=True)
    return current, record


def run_training(
    cfg: Config,
    run_name: str,
    *,
    dummy: bool = False,
    overfit: bool = False,
    resume: bool = False,
    stop_after: int | None = None,
    on_event: EventSink | None = None,
) -> TrainingSummary:
    started = time.perf_counter()
    run = prepare_run(cfg, run_name, dummy=dummy, overfit=overfit, stop_after=stop_after)
    cfg, directory = run.cfg, run.directory
    settings = cfg["training"]
    batch_size = settings["batch_size"]
    batches_per_epoch, total_steps = run.batches_per_epoch, run.total_steps
    max_seconds = run.max_seconds
    with (
        exclusive_lock(directory.parent / f".{run_name}.lock"),
        TrainingEvents(on_event, started) as events,
    ):
        progress = create_or_restore_run(run, resume=resume, stop_after=stop_after)
        if progress.finished_summary is not None:
            return progress.finished_summary
        state, history, previous_seconds = (
            progress.state,
            progress.history,
            progress.previous_seconds,
        )
        initial_step = int(state.step)
        events.previous_seconds = previous_seconds
        events.emit("started", directory=str(directory), step=initial_step, status="initializing")
        evaluation = prepare_evaluation(run)
        pending_evaluation = (
            history
            and history[-1]["step"] < int(state.step)
            and (int(state.step) % batches_per_epoch == 0 or int(state.step) == total_steps)
        )
        if not history or pending_evaluation:
            events.emit("phase", status="evaluating")
            state, record = measure(state, evaluation, batch_size)
            history.append(record)
            events.emit("evaluation", **record)
        update = make_update(run.optimizer, run.stats, run.sigma_n)
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
                        len(evaluation.x), batch_size, jax.random.fold_in(state.shuffle_key, epoch)
                    )
                    active_epoch = epoch
                indices, mask = epoch_order[batch], epoch_masks[batch]
                before = time.perf_counter()
                model, opt_state, key, loss, gradient_norm = update(
                    state.model,
                    state.opt_state,
                    state.noise_key,
                    evaluation.x[indices],
                    evaluation.u[indices],
                    mask,
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
                events.emit(
                    "update",
                    step=int(state.step),
                    batch_nll=loss,
                    gradient_norm=gradient_norm,
                    update_seconds=durations[-1],
                )
                check_target = (
                    overfit
                    and int(state.step) == overfit_settings(cfg, dummy)["target_check_steps"]
                )
                if (
                    int(state.step) % batches_per_epoch == 0
                    or int(state.step) == total_steps
                    or check_target
                ):
                    events.emit("phase", status="evaluating")
                    state, record = measure(state, evaluation, batch_size)
                    history.append(record)
                    events.emit("evaluation", **record)
                    if (
                        overfit
                        and int(state.step) >= overfit_settings(cfg, dummy)["target_check_steps"]
                        and record["train_nll"]
                        < evaluation.prior_train_nll
                        - overfit_settings(cfg, dummy).get("target_nll_gain", 0.0)
                    ):
                        stop_reason = "overfit_target"
                        break
                    if int(state.stale_epochs) >= settings["patience"]:
                        stop_reason = "early_stopping"
                        break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        summary: TrainingSummary = {
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
        events.emit("phase", status="saving", step=int(state.step))
        save_checkpoint(directory, state, run.identity, history, summary)
        write_json(directory / "summary.json", summary)
        events.emit("finished", status=stop_reason, step=int(state.step))
        return summary
