import csv
import hashlib
import io
import json
import os
import tempfile
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from kerr_sbi.config import Config, project_path
from kerr_sbi.obs_model import preprocess_pfm
from kerr_sbi.persistence import atomic_text, exclusive_lock, file_sha256, write_json
from kerr_sbi.prior import sample_prior
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.simulator import RenderError, render, template_sha256


def split_directory(cfg: Config, split: str) -> Path:
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    return project_path(cfg, "data") / split


def dataset_identity(cfg: Config, provenance: dict[str, Any]) -> dict[str, Any]:
    noise = cfg["observation"]["sigma_n"]
    if not isinstance(noise, (int, float)) or not np.isfinite(noise) or noise <= 0:
        raise ValueError("select a positive observation sigma_n at the M2 gate first")
    return {
        "schema_version": 1,
        "sampler": "numpy.PCG64; interleaved float64 uniform pairs (spin, cosine)",
        "numpy_version": np.__version__,
        "nullgeo": {key: provenance[key] for key in ("version", "source_commit", "binary_sha256")},
        "template_sha256": template_sha256(cfg),
        **{key: deepcopy(cfg[key]) for key in ("prior", "scene", "emission", "observation")},
    }


def parameter_csv(parameters: np.ndarray) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["idx", "a", "incl_deg", "cos_i"])
    for index, values in enumerate(parameters):
        writer.writerow([index, *(format(value, ".17g") for value in values)])
    return stream.getvalue()


def parse_parameters(text: str) -> np.ndarray:
    with io.StringIO(text) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["idx", "a", "incl_deg", "cos_i"]:
            raise ValueError("unexpected parameter table columns")
        rows = []
        for index, row in enumerate(reader):
            if row["idx"] != str(index):
                raise ValueError("parameter indices must be contiguous and ordered from zero")
            rows.append([float(row[key]) for key in ("a", "incl_deg", "cos_i")])
    parameters = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(parameters).all():
        raise ValueError("parameter table contains nonfinite values")
    return parameters


def read_parameters(path: Path) -> np.ndarray:
    return parse_parameters(path.read_text())


def validate_events(events: dict[int, dict[str, Any]], parameters: np.ndarray) -> None:
    for index, event in events.items():
        if type(index) is not int or not 0 <= index < len(parameters):
            raise ValueError("render ledger contains indices outside the parameter table")
        if event["status"] not in ("success", "failed") or not np.array_equal(
            [event["a"], event["incl_deg"]], parameters[index, :2]
        ):
            raise ValueError("render ledger parameters or status are inconsistent")


def read_events(path: Path, *, repair: bool = False) -> dict[int, dict[str, Any]]:
    events = {}
    if not path.exists():
        return events
    complete_bytes = 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            events[event["idx"]] = event
            complete_bytes += len(line)
    if repair and path.stat().st_size != complete_bytes:
        with path.open("r+b") as stream:
            stream.truncate(complete_bytes)
    return events


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_failures(path: Path, events: dict[int, dict[str, Any]]) -> None:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["idx", "a", "incl_deg", "attempts", "error", "stderr"])
    for index, event in sorted(events.items()):
        if event["status"] == "failed":
            writer.writerow(
                [index, *(event[key] for key in ("a", "incl_deg", "attempts", "error", "stderr"))]
            )
    atomic_text(path, stream.getvalue())


def checked_image(
    path: Path, cfg: Config, expected_hash: str | None = None
) -> tuple[np.ndarray, str]:
    width, height = cfg["scene"]["width"], cfg["scene"]["height"]
    expected_size = (
        len(f"PF\n{width} {height}\n-1.0\n".encode())
        + width * height * 3 * np.dtype(np.float32).itemsize
    )
    if path.stat().st_size != expected_size:
        raise ValueError(f"PFM has unexpected byte size: {path}")
    digest = file_sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"PFM hash differs from its successful render record: {path}")
    return preprocess_pfm(path, cfg), digest


def prepare_generation(
    cfg: Config, split: str, n: int, seed: int
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    if type(n) is not int or n < 1:
        raise ValueError("generation count must be a positive integer")
    parameters = sample_prior(n, seed, cfg)
    directory = split_directory(cfg, split)
    provenance = simulator_provenance(cfg)
    identity = dataset_identity(cfg, provenance)
    metadata_path = directory / "generation.json"
    params_path = directory / "params.csv"
    other = split_directory(cfg, "test" if split == "train" else "train") / "generation.json"
    if other.exists() and json.loads(other.read_text())["seed"] == seed:
        raise ValueError("train and test must use different prior seeds")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["split"] != split or metadata["seed"] != seed:
            raise ValueError("split or seed differs from the existing dataset")
        if metadata["identity"] != identity:
            raise ValueError(
                "dataset identity differs: simulator, prior, physics or observation model"
            )
        if params_path.exists():
            previous = read_parameters(params_path)
            if len(previous) > n:
                raise ValueError("cannot shrink an existing parameter table")
            if not np.array_equal(previous, parameters[: len(previous)]):
                raise ValueError("parameter table differs from the seeded prior draws")
            if len(previous) == n:
                return directory, parameters, metadata
        elif any((directory / "pfm").glob("*.pfm")):
            raise ValueError("rendered dataset is missing its parameter table")
    else:
        if params_path.exists() or any((directory / "pfm").glob("*.pfm")):
            raise ValueError("existing dataset lacks generation provenance")
        metadata = {
            "date_utc": datetime.now(UTC).isoformat(),
            "split": split,
            "seed": seed,
            "identity": identity,
            "nullgeo": provenance,
            "config": {key: value for key, value in cfg.items() if key != "project_root"},
            "parameter_columns": ["a", "incl_deg", "cos_i"],
        }
        write_json(metadata_path, metadata)
    atomic_text(params_path, parameter_csv(parameters))
    return directory, parameters, metadata


def generate_dataset(cfg: Config, split: str, n: int, seed: int) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    cfg["simulator"]["smoke_png"] = False
    with exclusive_lock(project_path(cfg, "data") / ".generation.lock"):
        directory, parameters, _ = prepare_generation(cfg, split, n, seed)
        ledger = directory / "renders.jsonl"
        events = read_events(ledger, repair=True)
        validate_events(events, parameters)
        failures = directory / "failures.csv"
        write_failures(failures, events)
        rendered = skipped = recovered = 0
        durations = []
        total_render_seconds = 0.0
        for index, (a, inclination, _) in enumerate(
            tqdm(parameters, desc=f"Generate {split}", unit="image")
        ):
            path = directory / "pfm" / f"{index:06d}.pfm"
            previous = events.get(index, {})
            try:
                _, digest = checked_image(path, cfg, previous.get("pfm_sha256"))
            except (OSError, ValueError):
                digest = None
            if digest is not None and previous.get("status") != "failed":
                skipped += 1
                if previous.get("status") == "success":
                    continue
                recovered += 1
                event = {
                    "idx": index,
                    "a": float(a),
                    "incl_deg": float(inclination),
                    "date_utc": datetime.now(UTC).isoformat(),
                    "status": "success",
                    "pfm_sha256": digest,
                    "render_seconds": None,
                    "attempts": 0,
                    "recovered_after_interruption": True,
                }
            else:
                started = time.perf_counter()
                attempts = 0
                while attempts < 2:
                    attempts += 1
                    try:
                        render(float(a), float(inclination), path.with_suffix(""), cfg)
                        _, digest = checked_image(path, cfg)
                        error = None
                        break
                    except (RenderError, OSError, ValueError) as exc:
                        error = exc
                elapsed = time.perf_counter() - started
                total_render_seconds += elapsed
                event = {
                    "idx": index,
                    "a": float(a),
                    "incl_deg": float(inclination),
                    "date_utc": datetime.now(UTC).isoformat(),
                    "attempts": attempts,
                    "render_seconds": elapsed,
                }
                if error is None:
                    event.update(status="success", pfm_sha256=digest)
                    durations.append(elapsed)
                    rendered += 1
                else:
                    event.update(
                        status="failed", error=str(error), stderr=getattr(error, "stderr", "")
                    )
                    tqdm.write(f"Failed index {index}: {error}")
            append_event(ledger, event)
            events[index] = event
            if event["status"] == "failed" or previous.get("status") == "failed":
                write_failures(failures, events)
        summary = {
            "split": split,
            "seed": seed,
            "n_requested": n,
            "n_rendered_this_run": rendered,
            "n_skipped": skipped,
            "n_recovered": recovered,
            "n_failed": sum(event["status"] == "failed" for event in events.values()),
            "render_seconds_this_run": total_render_seconds,
            "median_successful_render_seconds": float(np.median(durations)) if durations else None,
            "date_utc": datetime.now(UTC).isoformat(),
        }
        write_json(directory / "generation_summary.json", summary)
        return summary


def preprocess_dataset(cfg: Config, split: str) -> dict[str, Any]:
    directory = split_directory(cfg, split)
    with exclusive_lock(directory / ".preprocess.lock"):
        generation = json.loads((directory / "generation.json").read_text())
        if generation["split"] != split or generation["identity"] != dataset_identity(
            cfg, generation["nullgeo"]
        ):
            raise ValueError("preprocessing configuration differs from the generation identity")
        params_path = directory / "params.csv"
        params_bytes = params_path.read_bytes()
        parameters = parse_parameters(params_bytes.decode())
        if not np.array_equal(parameters, sample_prior(len(parameters), generation["seed"], cfg)):
            raise ValueError("parameter table differs from the seeded prior draws")
        events = {
            index: event
            for index, event in read_events(directory / "renders.jsonl").items()
            if not isinstance(index, int) or index < len(parameters)
        }
        validate_events(events, parameters)
        indices, images, hashes, invalid, missing = [], [], {}, {}, []
        for index in tqdm(range(len(parameters)), desc=f"Preprocess {split}", unit="image"):
            path = directory / "pfm" / f"{index:06d}.pfm"
            if not path.exists():
                missing.append(index)
                continue
            event = events.get(index, {})
            if event.get("status") == "failed":
                invalid[index] = "latest render attempt failed; rerun generation"
                continue
            try:
                image, digest = checked_image(path, cfg, event.get("pfm_sha256"))
            except (OSError, ValueError) as exc:
                invalid[index] = str(exc)
                continue
            indices.append(index)
            images.append(image)
            hashes[str(index)] = digest
        idx = np.asarray(indices, dtype=np.int64)
        shape = (cfg["scene"]["height"], cfg["scene"]["width"])
        x = np.stack(images) if images else np.empty((0, *shape), dtype=np.float32)
        arrays = {"x.npy": x, "theta.npy": parameters[idx, :2], "idx.npy": idx}
        failed = sorted(index for index, event in events.items() if event["status"] == "failed")
        metadata = {
            "date_utc": datetime.now(UTC).isoformat(),
            "split": split,
            "seed": generation["seed"],
            "nullgeo": generation["nullgeo"],
            "template_sha256": generation["identity"]["template_sha256"],
            "identity": generation["identity"],
            "generation_sha256": file_sha256(directory / "generation.json"),
            "params_sha256": hashlib.sha256(params_bytes).hexdigest(),
            "prior": cfg["prior"],
            "observation": cfg["observation"],
            "n_requested": len(parameters),
            "n_rendered": len(indices),
            "n_failed": len(failed),
            "failed_indices": failed,
            "missing_indices": missing,
            "invalid_indices": invalid,
            "theta_columns": ["a", "incl_deg"],
            "cos_i_location": "params.csv, indexed by idx.npy",
            "measurement_space": (
                "linear luminance, zero-padded PSF, flux normalization; no noise or asinh"
            ),
            "pfm_sha256": hashes,
            "publication": (
                "arrays replaced before meta.json; verify array hashes before reading a snapshot"
            ),
        }
        with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=directory) as temporary:
            stage = Path(temporary)
            for name, values in arrays.items():
                np.save(stage / name, values, allow_pickle=False)
            metadata["arrays"] = {
                name: {
                    "sha256": file_sha256(stage / name),
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                }
                for name, values in arrays.items()
            }
            write_json(stage / "meta.json", metadata)
            for name in (*arrays, "meta.json"):
                (stage / name).replace(directory / name)
        return metadata


def load_preprocessed(cfg: Config, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = split_directory(cfg, split)
    metadata = json.loads((directory / "meta.json").read_text())
    arrays = {}
    for name in ("x.npy", "theta.npy", "idx.npy"):
        path = directory / name
        with path.open("rb") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != metadata["arrays"][name]["sha256"]
            ):
                raise ValueError(
                    "preprocessed snapshot is incomplete or corrupt; rerun preprocessing"
                )
            stream.seek(0)
            values = np.load(stream, allow_pickle=False)
        if (
            list(values.shape) != metadata["arrays"][name]["shape"]
            or str(values.dtype) != metadata["arrays"][name]["dtype"]
        ):
            raise ValueError("preprocessed array shape or dtype differs from its metadata")
        arrays[name.removesuffix(".npy")] = values
    return arrays, metadata
