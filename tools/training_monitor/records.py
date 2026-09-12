import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


def contained(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path is outside the selected project")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError:
        return []
    events = []
    for line in lines:
        if not line.endswith(b"\n"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("schema_version") == 1:
            if event.get("kind") == "started":
                boundary = event.get("step", 0)
                events = [e for e in events if e.get("step", 0) <= boundary]
            events.append(event)
    return events


class Records:
    def __init__(self, root: Path, runs: Path | None = None, data: Path | None = None):
        self.root = root.resolve()
        self.runs = contained(self.root, runs or self.root / "runs")
        self.data = contained(self.root, data or self.root / "data")
        self.cached_storage: dict[str, Any] = {}
        self.storage_updated = 0.0

    def storage(self) -> dict[str, Any]:
        if time.monotonic() - self.storage_updated > 30:
            usage = shutil.disk_usage(self.root)
            self.cached_storage = {
                "free": usage.free,
                "total": usage.total,
                "data": file_bytes(self.data),
                "runs": file_bytes(self.runs),
                "results": file_bytes(self.root / "results"),
                "planned_dataset": 13000 * (64 * 64 * 16 + 14),
            }
            self.storage_updated = time.monotonic()
        return self.cached_storage

    def run(self, identifier: str) -> dict[str, Any]:
        directory = contained(self.runs, self.runs / identifier)
        identity = read_json(contained(self.runs, directory / "run.json"))
        if not identity:
            raise FileNotFoundError("run not found")
        pointer = read_json(contained(self.runs, directory / "latest.json"))
        checkpoint = {}
        checkpoint_timestamp = 0.0
        if pointer.get("directory"):
            target = contained(directory, directory / pointer["directory"])
            metadata_path = contained(directory, target / "metadata.json")
            checkpoint = read_json(metadata_path)
            if checkpoint:
                checkpoint_timestamp = metadata_path.stat().st_mtime
        summary = checkpoint.get("summary") or read_json(
            contained(directory, directory / "summary.json")
        )
        history = {r["step"]: r for r in checkpoint.get("history", [])}
        events = read_events(contained(directory, directory / "events.jsonl"))
        for event in events:
            if event.get("kind") == "evaluation":
                history[event["step"]] = event
        latest = events[-1] if events else {}
        step = latest.get("step", summary.get("step", 0))
        elapsed = latest.get("elapsed_seconds", summary.get("elapsed_seconds_total"))
        status = summary.get("stop_reason", "unavailable")
        if events:
            status = latest.get("status", "running")
            if status in ("running", "initializing", "evaluating", "saving"):
                if time.time() - latest.get("timestamp", 0) > 30:
                    status = "unconfirmed"
            if checkpoint_timestamp > latest.get("timestamp", 0) and summary.get("step", 0) >= step:
                step = summary["step"]
                status = summary["stop_reason"]
                elapsed = summary.get("elapsed_seconds_total")
        settings = identity.get("config", {}).get("training", {})
        train_count = len(identity.get("train_indices", []))
        batch = settings.get("batch_size", 1)
        batches = max(1, math.ceil(train_count / batch))
        updates = [e for e in events if e.get("kind") == "update"]
        last_update = updates[-1] if updates else {}
        evaluations = sorted(history.values(), key=lambda r: r["step"])
        current = evaluations[-1] if evaluations else summary.get("last_evaluation", {})
        recent_evaluations = evaluations[-4:]
        recent_rates = [
            (right["timestamp"] - left["timestamp"]) / (right["step"] - left["step"])
            for left, right in zip(recent_evaluations, recent_evaluations[1:], strict=False)
            if right["step"] > left["step"]
            and "timestamp" in left
            and "timestamp" in right
            and left.get("attempt") == right.get("attempt")
            and right["timestamp"] > left["timestamp"]
        ]
        remaining = None
        if recent_rates and status in ("running", "evaluating"):
            remaining = float(np.median(recent_rates)) * max(
                0, identity.get("total_steps", 0) - step
            )
        return {
            "id": identifier,
            "name": directory.name,
            "dummy": identity.get("dummy", False),
            "overfit": identity.get("overfit", False),
            "status": status,
            "step": step,
            "total_steps": identity.get("total_steps", 0),
            "epoch": step / batches,
            "batches_per_epoch": batches,
            "train_count": train_count,
            "validation_count": len(identity.get("validation_indices", [])),
            "noise": identity.get("effective_sigma_n"),
            "device": identity.get("device", "unknown"),
            "settings": settings,
            "normalization": identity.get("normalization", {}),
            "versions": identity.get("versions", {}),
            "dataset": identity.get("dataset", {}),
            "history": evaluations,
            "updates": updates[-1000:],
            "current": current,
            "best_validation": min((r["validation_nll"] for r in evaluations), default=None),
            "last_update": last_update,
            "elapsed": elapsed,
            "remaining_seconds": remaining,
            "checkpoint_step": checkpoint.get("summary", {}).get("step"),
            "checkpoint": pointer.get("directory"),
            "events_available": bool(events),
            "last_seen": latest.get("timestamp"),
            "error": latest.get("error"),
            "modified": (directory / "run.json").stat().st_mtime,
        }

    def datasets(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.data.rglob("meta.json")):
            if not path.resolve().is_relative_to(self.data):
                continue
            metadata = read_json(path)
            if "n_requested" not in metadata or not (path.parent / "x.npy").is_file():
                continue
            dummy = metadata.get("dataset_kind") == "dummy"
            rendered = metadata.get("n_examples" if dummy else "n_rendered", 0)
            result.append(
                {
                    "id": str(path.parent.relative_to(self.data)),
                    "dummy": dummy,
                    "split": metadata.get("split"),
                    "count": rendered,
                    "requested": metadata["n_requested"],
                    "missing": len(metadata.get("missing_indices", [])),
                    "failed": metadata.get("n_failed", 0),
                    "seed": metadata.get("seed"),
                    "noise": metadata.get("observation", {}).get("sigma_n"),
                    "simulator": metadata.get("nullgeo"),
                }
            )
        return result

    def overview(self) -> dict[str, Any]:
        runs, errors = [], []
        for path in self.runs.rglob("run.json"):
            identifier = str(path.parent.relative_to(self.runs))
            try:
                run = self.run(identifier)
                runs.append(
                    {
                        key: run[key]
                        for key in (
                            "id",
                            "name",
                            "dummy",
                            "overfit",
                            "status",
                            "step",
                            "total_steps",
                            "modified",
                        )
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                errors.append(identifier)
        runs.sort(key=lambda r: r["modified"], reverse=True)
        return {
            "runs": runs,
            "datasets": self.datasets(),
            "storage": self.storage(),
            "unreadable_runs": errors,
            "timestamp": time.time(),
        }

    def preview(self, identifier: str, index: int) -> dict[str, Any]:
        directory = contained(self.data, self.data / identifier)
        meta = read_json(contained(self.data, directory / "meta.json"))
        if not meta or meta.get("split") != "train":
            raise ValueError("only training images are available for preview")
        paths = [contained(self.data, directory / f"{name}.npy") for name in ("x", "theta", "idx")]
        images, theta, indices = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
        if images.ndim != 3 or images.shape[1:] != (64, 64) or not 0 <= index < len(images):
            raise ValueError("image index or shape is invalid")
        values = np.asarray(images[index], dtype=np.float64)
        if not np.isfinite(values).all() or values.min() < 0 or values.max() <= 0:
            raise ValueError("preview contains invalid image values")
        stretched = np.arcsinh(values / (0.1 * values.max())) / np.arcsinh(10)
        return {
            "pixels": np.rint(stretched * 255).astype(np.uint8).ravel().tolist(),
            "idx": int(indices[index]),
            "spin": float(theta[index, 0]),
            "inclination": float(theta[index, 1]),
            "peak": float(values.max()),
            "flux": float(values.sum()),
        }
