import json
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import equinox as eqx
import jax
import optax

from kerr_sbi.model import Posterior
from kerr_sbi.persistence import file_sha256, write_json


class TrainState(eqx.Module):
    model: Posterior
    best_model: Posterior
    opt_state: optax.OptState
    noise_key: jax.Array
    shuffle_key: jax.Array
    step: jax.Array
    best_validation: jax.Array
    stale_epochs: jax.Array


class EvaluationRecord(TypedDict):
    step: int
    train_nll: float
    validation_nll: float
    prior_train_nll: float
    prior_validation_nll: float


class TrainingSummary(TypedDict):
    run: str
    dummy: bool
    overfit: bool
    stop_reason: str
    step: int
    updates_this_invocation: int
    elapsed_seconds_total: float
    first_update_seconds_including_compile: float | None
    median_later_update_seconds: float | None
    max_gradient_norm: float | None
    last_batch_nll: float | None
    best_validation_nll: float
    last_evaluation: EvaluationRecord
    checkpoint_root: str


class CheckpointMetadata(TypedDict):
    identity: dict[str, Any]
    history: list[EvaluationRecord]
    summary: TrainingSummary
    files: dict[str, str]


def save_checkpoint(
    directory: Path,
    state: TrainState,
    identity: dict[str, Any],
    history: list[EvaluationRecord],
    summary: TrainingSummary,
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


def read_checkpoint(directory: Path, identity: dict[str, Any]) -> tuple[Path, CheckpointMetadata]:
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
    return root, metadata


def load_checkpoint(
    directory: Path, template: TrainState, identity: dict[str, Any]
) -> tuple[TrainState, CheckpointMetadata]:
    root, metadata = read_checkpoint(directory, identity)
    state = eqx.tree_deserialise_leaves(root / "state.eqx", template)
    return state, metadata
