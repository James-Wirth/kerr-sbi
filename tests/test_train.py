import json
from copy import deepcopy
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerr_sbi import train
from kerr_sbi.config import Config
from kerr_sbi.dummy import build_dummy_dataset
from kerr_sbi.train import (
    effective_config,
    initialize_state,
    load_checkpoint,
    optimizer_for,
    posterior_check,
    run_training,
)


def assert_same_arrays(left: object, right: object) -> None:
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
        if eqx.is_array(a):
            np.testing.assert_array_equal(a, b)


def test_resume_matches_uninterrupted_training_and_checkpoint_predictions(
    small_model_cfg: Config, tmp_path: Path
) -> None:
    cfg = small_model_cfg
    cfg["project_root"] = tmp_path
    cfg["dummy_dataset"].update(train_count=12, test_count=4)
    cfg["dummy_training"].update(batch_size=2, warmup_steps=1, max_steps=7, posterior_samples=3)
    build_dummy_dataset(cfg)
    original = deepcopy(cfg)
    paused = run_training(cfg, "resumed", dummy=True, stop_after=4)
    assert paused["step"] == 4 and paused["stop_reason"] == "paused"
    resumed = run_training(cfg, "resumed", dummy=True, resume=True)
    clean = run_training(cfg, "clean", dummy=True)
    assert resumed["step"] == clean["step"] == 7
    completed_resume = run_training(cfg, "clean", dummy=True, resume=True)
    assert completed_resume["updates_this_invocation"] == 0
    assert cfg == original
    effective = effective_config(cfg, dummy=True, overfit=False)
    template = initialize_state(effective, optimizer_for(effective, 7))
    states = []
    for name in ("resumed", "clean"):
        directory = tmp_path / "runs" / "dummy" / name
        identity = json.loads((directory / "run.json").read_text())
        state, metadata = load_checkpoint(directory, template, identity)
        assert identity["dummy"] and identity["dataset"]["scientific_use"] is False
        assert len(identity["train_indices"]) == 10 and len(identity["validation_indices"]) == 2
        pointer = json.loads((directory / "latest.json").read_text())
        best = eqx.tree_deserialise_leaves(
            directory / pointer["directory"] / "best.eqx", template.model
        )
        z, u = jnp.zeros((1, 1, 64, 64)), jnp.zeros((1, 2))
        np.testing.assert_array_equal(best.log_prob(z, u), state.best_model.log_prob(z, u))
        assert np.isfinite(metadata["summary"]["max_gradient_norm"])
        states.append(state)
    assert_same_arrays(*states)
    for name in ("embedding", "flow"):
        initial = jax.tree_util.tree_leaves(
            eqx.filter(getattr(template.model, name), eqx.is_inexact_array)
        )
        trained = jax.tree_util.tree_leaves(
            eqx.filter(getattr(states[0].model, name), eqx.is_inexact_array)
        )
        assert any(not np.array_equal(a, b) for a, b in zip(initial, trained, strict=True))
    report = posterior_check(cfg, "clean")
    assert report["samples_shape"] == [3, 4, 2] and report["ranks_shape"] == [4, 2]
    assert report["scientific_calibration"] is False
    changed = deepcopy(cfg)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="checkpoint config"):
        run_training(changed, "resumed", dummy=True, resume=True)
    directory = tmp_path / "runs/dummy/resumed"
    pointer = json.loads((directory / "latest.json").read_text())
    path = directory / pointer["directory"] / "state.eqx"
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="array hash"):
        load_checkpoint(directory, template, json.loads((directory / "run.json").read_text()))


def test_short_schedule_and_explicit_debug_mode(cfg: Config) -> None:
    with pytest.raises(ValueError, match="warmup"):
        optimizer_for(cfg, 20)
    with pytest.raises(ValueError, match="explicit dummy"):
        effective_config(cfg, dummy=False, overfit=True)
    dev = effective_config(cfg, dummy=True, overfit=False)
    assert dev["training"]["warmup_steps"] == 5
    assert dev["training"]["batch_size"] == 8
    optimizer_for(dev, 20)
    debug = effective_config(cfg, dummy=True, overfit=True)
    assert debug["training"]["max_steps"] == 200
    assert debug["observation"]["sigma_n"] == cfg["observation"]["sigma_n"] == 20


@pytest.mark.parametrize("stop_after", [4, 5])
def test_resume_across_shuffle_boundary_preserves_losses_and_stopping(
    small_model_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_after: int
) -> None:
    cfg = small_model_cfg
    cfg["project_root"] = tmp_path
    cfg["dummy_dataset"].update(train_count=11, test_count=2)
    cfg["dummy_training"].update(batch_size=2, warmup_steps=1, max_steps=12)
    cfg["training"].update(learning_rate=0.0, patience=2)
    build_dummy_dataset(cfg)
    losses = []
    make_update = train.make_update

    def recording_update(*args):
        update = make_update(*args)

        def record(*batch):
            result = update(*batch)
            losses.append((float(result[3]), float(result[4])))
            return result

        return record

    monkeypatch.setattr(train, "make_update", recording_update)
    paused = run_training(cfg, "resumed", dummy=True, stop_after=stop_after)
    assert paused["step"] == stop_after and paused["stop_reason"] == "paused"
    resumed = run_training(cfg, "resumed", dummy=True, resume=True)
    resumed_losses = losses.copy()
    losses.clear()
    clean = run_training(cfg, "clean", dummy=True)
    assert resumed["step"] == clean["step"] == 10
    assert resumed["stop_reason"] == clean["stop_reason"] == "early_stopping"
    assert resumed["last_batch_nll"] == clean["last_batch_nll"]
    np.testing.assert_array_equal(resumed_losses, losses)
    effective = effective_config(cfg, dummy=True, overfit=False)
    template = initialize_state(effective, optimizer_for(effective, 12))
    states, histories = [], []
    for name in ("resumed", "clean"):
        directory = tmp_path / "runs/dummy" / name
        identity = json.loads((directory / "run.json").read_text())
        state, metadata = load_checkpoint(directory, template, identity)
        assert len(identity["train_indices"]) == 9
        assert int(state.stale_epochs) == 2
        states.append(state)
        histories.append(metadata["history"])
    assert_same_arrays(*states)
    assert histories[0] == histories[1]
    assert [record["step"] for record in histories[0]] == [0, 5, 10]
