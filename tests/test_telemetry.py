import json
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from kerr_sbi.config import Config
from kerr_sbi.dummy import build_dummy_dataset
from kerr_sbi.experiments import experiment_config, preflight
from kerr_sbi.telemetry import JsonlEvents, TrainingEvents
from kerr_sbi.train import (
    effective_config,
    initialize_state,
    load_checkpoint,
    optimizer_for,
    run_training,
)


def test_events_preserve_optimizer_rng_and_resume(small_model_cfg: Config, tmp_path: Path) -> None:
    cfg = small_model_cfg
    cfg["project_root"] = tmp_path
    cfg["dummy_dataset"].update(train_count=11, test_count=2)
    cfg["dummy_training"].update(batch_size=2, warmup_steps=1, max_steps=7)
    build_dummy_dataset(cfg)
    records = []
    writer = JsonlEvents(interval_seconds=0)

    def sink(event: dict) -> None:
        records.append(event)
        writer(event)

    run_training(cfg, "observed", dummy=True, stop_after=4, on_event=sink)
    run_training(cfg, "observed", dummy=True, resume=True, on_event=sink)
    run_training(cfg, "plain", dummy=True)
    steps = [event["step"] for event in records if event["kind"] == "update"]
    assert steps == list(range(1, 8))
    assert [e["step"] for e in records if e["kind"] == "started"] == [0, 4]
    assert [e["status"] for e in records if e["kind"] == "finished"] == ["paused", "complete"]
    assert [e["step"] for e in records if e["kind"] == "evaluation"] == [0, 5, 7]
    effective = effective_config(cfg, dummy=True, overfit=False)
    template = initialize_state(effective, optimizer_for(effective, 7))
    states = []
    for name in ("observed", "plain"):
        directory = tmp_path / "runs/dummy" / name
        identity = json.loads((directory / "run.json").read_text())
        state, _ = load_checkpoint(directory, template, identity)
        states.append(state)
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        if eqx.is_array(left):
            np.testing.assert_array_equal(left, right)
    path = tmp_path / "runs/dummy/observed/events.jsonl"
    saved = path.read_bytes()
    run_training(cfg, "observed", dummy=True, resume=True, on_event=sink)
    assert path.read_bytes() == saved


def test_observer_failure_does_not_interrupt_training() -> None:
    def failed(event: dict) -> None:
        raise OSError("disk unavailable")

    with TrainingEvents(failed, time.perf_counter()) as events:
        with pytest.warns(RuntimeWarning, match="telemetry disabled"):
            events.emit("started")
        events.emit("update", step=1)
        assert events.sink is None
    captured = []
    with pytest.raises(ValueError, match="training failure"):
        with TrainingEvents(captured.append, time.perf_counter()) as events:
            events.emit("started")
            events.emit("update", step=9)
            raise ValueError("training failure")
    assert captured[-1]["status"] == "failed" and captured[-1]["step"] == 9


def test_jsonl_throttling_and_partial_record_recovery(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"partial":')
    writer = JsonlEvents(1000)
    with TrainingEvents(writer, time.perf_counter()) as events:
        events.emit("started", directory=str(tmp_path))
        events.emit("update", step=1)
        events.emit("update", step=2)
        events.emit("evaluation", step=2, validation_nll=1)
        events.emit("finished", status="complete")
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines[1:]]
    assert [r["kind"] for r in records] == ["started", "update", "evaluation", "finished"]


def test_local_profile_and_preflight_are_read_only(cfg: Config, tmp_path: Path) -> None:
    cfg["project_root"] = tmp_path
    original_batch = cfg["training"]["batch_size"]
    smoke = experiment_config(cfg, "local-smoke")
    assert cfg["training"]["batch_size"] == original_batch == 256
    assert smoke["training"]["max_steps"] == 40
    assert smoke["observation"] == cfg["observation"]
    assert smoke["training"]["conv_channels"] == cfg["training"]["conv_channels"]
    build_dummy_dataset(cfg)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    report = preflight(cfg, dummy=True, overfit=False)
    assert report["total_steps"] == 20 and report["sigma_n"] == 20
    assert report["train_rows"] == 57 and report["validation_rows"] == 7
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
