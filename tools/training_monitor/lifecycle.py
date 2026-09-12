import fcntl
import json
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def lock(path: Path) -> Iterator[None]:
    if path.is_symlink():
        raise ValueError("symlinked locks are not supported")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "This run or its trash is busy. Try again after training stops."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def is_active(directory: Path) -> bool:
    return any(
        is_locked(directory.parent / f".{directory.name}{suffix}")
        for suffix in (".lock", ".job.lock")
    )


def is_locked(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink():
        return True
    with path.open("r") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def regular_directory(root: Path, path: Path) -> Path:
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("symlinked run directories are not supported")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path is outside the run directory")
    return path


class RunTrash:
    def __init__(self, root: Path, runs: Path, data: Path):
        self.root = root.resolve()
        self.runs = runs.resolve()
        self.data = data.resolve()
        self.directory = self.runs / ".trash"

    def source(self, identifier: str) -> Path:
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", identifier
        ):
            raise ValueError("invalid run identifier")
        path = regular_directory(self.runs, self.runs / identifier)
        if path == self.data or self.data.is_relative_to(path):
            raise ValueError("shared datasets cannot be removed")
        return path

    def validate_run(self, path: Path) -> dict[str, Any]:
        metadata_path = path / "run.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("a regular run.json is required")
        identity = json.loads(metadata_path.read_text())
        if not isinstance(identity, dict) or not isinstance(identity.get("config"), dict):
            raise ValueError("invalid run metadata")
        dataset = identity["config"].get("paths", {}).get("data")
        if dataset:
            dataset_path = (self.root / dataset).resolve()
            if dataset_path == path or dataset_path.is_relative_to(path):
                raise ValueError("this run contains a dataset; move the dataset out before cleanup")
        if any(path.rglob("params.csv")) or any(path.rglob("generation.json")):
            raise ValueError("this directory contains simulation data")
        if any(item != metadata_path for item in path.rglob("run.json")):
            raise ValueError("a run containing other runs cannot be removed")
        return identity

    def trash(self, identifier: str) -> dict[str, Any]:
        source = self.source(identifier)
        regular_directory(self.runs, self.directory)
        with (
            lock(self.runs / ".trash.lock"),
            lock(source.parent / f".{source.name}.job.lock"),
            lock(source.parent / f".{source.name}.lock"),
        ):
            self.validate_run(source)
            entry = self.directory / uuid.uuid4().hex
            entry.mkdir(parents=True)
            manifest = {"id": entry.name, "original": identifier, "deleted_at": time.time()}
            (entry / "manifest.json").write_text(json.dumps(manifest) + "\n")
            try:
                source.rename(entry / "run")
            except OSError:
                (entry / "manifest.json").unlink()
                entry.rmdir()
                raise
            return manifest

    def entry(self, identifier: str) -> tuple[Path, dict[str, Any]]:
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("invalid trash identifier")
        entry = regular_directory(self.runs, self.directory / identifier)
        manifest_path = entry / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("symlinked trash manifests are not supported")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("id") != identifier:
            raise ValueError("trash manifest differs")
        self.source(manifest["original"])
        regular_directory(self.runs, entry / "run")
        self.validate_run(entry / "run")
        return entry, manifest

    def entries(self) -> list[dict[str, Any]]:
        regular_directory(self.runs, self.directory)
        result = []
        if not self.directory.exists():
            return result
        for path in self.directory.iterdir():
            try:
                _, manifest = self.entry(path.name)
                manifest["bytes"] = sum(
                    item.stat().st_size
                    for item in path.rglob("*")
                    if item.is_file() and not item.is_symlink()
                )
                result.append(manifest)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda item: item["deleted_at"], reverse=True)

    def restore(self, identifier: str) -> dict[str, Any]:
        with lock(self.runs / ".trash.lock"):
            entry, manifest = self.entry(identifier)
            destination = self.source(manifest["original"])
            with (
                lock(destination.parent / f".{destination.name}.job.lock"),
                lock(destination.parent / f".{destination.name}.lock"),
            ):
                if destination.exists():
                    raise FileExistsError("a run with this name already exists")
                destination.parent.mkdir(parents=True, exist_ok=True)
                (entry / "run").rename(destination)
                (entry / "manifest.json").unlink()
                entry.rmdir()
            return manifest

    def purge(self, identifiers: list[str]) -> dict[str, int]:
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("select distinct trash entries to permanently delete")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError("safe recursive deletion is unavailable on this platform")
        with lock(self.runs / ".trash.lock"):
            entries = [self.entry(identifier)[0] for identifier in identifiers]
            for entry in entries:
                shutil.rmtree(entry)
        return {"deleted": len(entries)}
