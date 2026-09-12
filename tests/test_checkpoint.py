import json
from pathlib import Path

import pytest

from kerr_sbi.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from kerr_sbi.config import Config
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.train import initialize_state, optimizer_for


@pytest.fixture
def checkpoint(small_model_cfg: Config, tmp_path: Path) -> tuple:
    state = initialize_state(small_model_cfg, optimizer_for(small_model_cfg, 1000))
    identity = {"source_sha256": {"train.py": "historical"}}
    record = {
        "step": 0,
        "train_nll": 3.0,
        "validation_nll": 4.0,
        "prior_train_nll": 5.0,
        "prior_validation_nll": 6.0,
    }
    summary = {
        "run": "fixture",
        "dummy": True,
        "overfit": False,
        "stop_reason": "paused",
        "step": 0,
        "updates_this_invocation": 0,
        "elapsed_seconds_total": 0.0,
        "first_update_seconds_including_compile": None,
        "median_later_update_seconds": None,
        "max_gradient_norm": None,
        "last_batch_nll": None,
        "best_validation_nll": 4.0,
        "last_evaluation": record,
        "checkpoint_root": str(tmp_path),
    }
    save_checkpoint(tmp_path, state, identity, [record], summary)
    return tmp_path, state, identity


@pytest.mark.parametrize(
    ("name", "message"),
    [("metadata.json", "metadata hash"), ("state.eqx", "array hash"), ("best.eqx", "array hash")],
)
def test_checkpoint_rejects_corruption(checkpoint: tuple, name: str, message: str) -> None:
    directory, template, identity = checkpoint
    root, _ = read_checkpoint(directory, identity)
    path = root / name
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match=message):
        load_checkpoint(directory, template, identity)


def test_checkpoint_rejects_missing_manifest_entry(checkpoint: tuple) -> None:
    directory, template, identity = checkpoint
    root, metadata = read_checkpoint(directory, identity)
    del metadata["files"]["best.eqx"]
    write_json(root / "metadata.json", metadata)
    write_json(
        directory / "latest.json",
        {"directory": root.name, "metadata_sha256": file_sha256(root / "metadata.json")},
    )
    with pytest.raises(ValueError, match="manifest is incomplete"):
        load_checkpoint(directory, template, identity)


def test_checkpoint_rejects_directory_traversal(checkpoint: tuple) -> None:
    directory, template, identity = checkpoint
    pointer = json.loads((directory / "latest.json").read_text())
    pointer["directory"] = "../checkpoint_00000000"
    write_json(directory / "latest.json", pointer)
    with pytest.raises(ValueError, match="invalid checkpoint directory"):
        load_checkpoint(directory, template, identity)


def test_checkpoint_requires_exact_identity_and_preserves_recorded_hashes(
    checkpoint: tuple,
) -> None:
    directory, template, identity = checkpoint
    changed = {"source_sha256": {"train.py": "refactored"}}
    with pytest.raises(ValueError, match="checkpoint config"):
        load_checkpoint(directory, template, changed)
    _, metadata = load_checkpoint(directory, template, identity)
    assert metadata["identity"] == identity
