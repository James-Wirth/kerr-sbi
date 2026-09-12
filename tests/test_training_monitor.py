import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

MONITOR = Path(__file__).resolve().parents[1] / "tools/training_monitor/records.py"
spec = importlib.util.spec_from_file_location("monitor_records", MONITOR)
records_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records_module)
Records = records_module.Records


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def monitor(tmp_path: Path) -> tuple:
    directory = tmp_path / "runs/example"
    identity = {
        "config": {"training": {"batch_size": 4}},
        "train_indices": list(range(8)),
        "validation_indices": [8, 9],
        "total_steps": 20,
        "dummy": False,
    }
    write_json(directory / "run.json", identity)
    write_json(directory / "latest.json", {"directory": "checkpoint_00000004"})
    write_json(
        directory / "checkpoint_00000004/metadata.json",
        {
            "summary": {"step": 4, "stop_reason": "paused"},
            "history": [{"step": 0, "validation_nll": 3}, {"step": 4, "validation_nll": 2}],
        },
    )
    return Records(tmp_path), directory


def test_archived_and_live_runs_and_resume_rollback(monitor: tuple) -> None:
    reader, directory = monitor
    run = reader.run("example")
    assert run["status"] == "paused" and run["step"] == 4
    assert run["best_validation"] == 2 and not run["events_available"]
    events = [
        {"kind": "started", "step": 4},
        {"kind": "evaluation", "step": 8, "validation_nll": 0.5},
        {"kind": "started", "step": 4},
        {"kind": "update", "step": 5, "status": "running"},
    ]
    path = directory / "events.jsonl"
    path.write_text(
        "".join(
            json.dumps({"schema_version": 1, "timestamp": time.time(), **e}) + "\n" for e in events
        )
        + '{"unfinished":'
    )
    run = reader.run("example")
    assert run["step"] == 5 and run["status"] == "running"
    assert run["best_validation"] == 2
    assert [r["step"] for r in run["history"]] == [0, 4]
    path.write_text(
        json.dumps(
            {"schema_version": 1, "kind": "update", "step": 5, "status": "running", "timestamp": 1}
        )
        + "\n"
    )
    assert reader.run("example")["status"] == "unconfirmed"
    metadata_path = directory / "checkpoint_00000004/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["summary"] = {"step": 7, "stop_reason": "complete"}
    write_json(metadata_path, metadata)
    run = reader.run("example")
    assert run["status"] == "complete" and run["step"] == 7


def test_monitor_restricts_paths_and_reports_corrupt_runs(monitor: tuple, tmp_path: Path) -> None:
    reader, directory = monitor
    with pytest.raises(ValueError, match="outside"):
        reader.run("../../private")
    write_json(directory / "latest.json", {"directory": "../../"})
    assert reader.overview()["unreadable_runs"] == ["example"]
    outside = tmp_path / "outside.json"
    write_json(outside, {"secret": "not served"})
    link = tmp_path / "runs/link"
    link.mkdir()
    (link / "run.json").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        reader.run("link")


def test_dataset_counts_and_preview_do_not_touch_training_or_test_files(tmp_path: Path) -> None:
    for split in ("train", "test"):
        directory = tmp_path / "data/dummy" / split
        write_json(
            directory / "meta.json",
            {
                "dataset_kind": "dummy",
                "split": split,
                "n_requested": 2,
                "n_rendered": 0,
                "n_examples": 2,
            },
        )
        np.save(directory / "x.npy", np.ones((2, 64, 64), dtype=np.float32))
        np.save(directory / "theta.npy", np.array([[0.1, 20], [0.8, 60]]))
        np.save(directory / "idx.npy", np.array([0, 1]))
    reader = Records(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    assert [d["count"] for d in reader.datasets()] == [2, 2]
    preview = reader.preview("dummy/train", 1)
    assert preview["spin"] == 0.8 and preview["flux"] == 4096
    assert len(preview["pixels"]) == 4096
    with pytest.raises(ValueError, match="only training"):
        reader.preview("dummy/test", 0)
    with pytest.raises(ValueError, match="index"):
        reader.preview("dummy/train", 3)
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
