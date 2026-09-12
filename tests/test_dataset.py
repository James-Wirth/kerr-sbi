import csv
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from kerr_sbi import dataset
from kerr_sbi.config import Config, project_path
from kerr_sbi.prior import sample_prior
from kerr_sbi.simulator import RenderError


@pytest.fixture
def generation_setup(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    cfg["paths"]["scene_template"] = str(project_path(cfg, "scene_template"))
    cfg["project_root"] = tmp_path
    cfg["scene"].update(width=2, height=2)
    calls = []
    provenance = {"version": "0.2.0", "source_commit": "revision", "binary_sha256": "binary"}

    def fake_render(a: float, inclination: float, stem: Path, settings: Config) -> Path:
        table = dataset.read_parameters(stem.parent.parent / "params.csv")
        np.testing.assert_array_equal(table[int(stem.name), :2], [a, inclination])
        calls.append((int(stem.name), a, inclination))
        y = np.array([[1 + a, 2 + inclination / 90], [3, 4]], dtype=np.float32)
        rgb = np.repeat(y[..., None], 3, axis=-1)
        path = stem.with_suffix(".pfm")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PF\n2 2\n-1.0\n" + rgb[::-1].astype("<f4").tobytes())
        return path

    monkeypatch.setattr(dataset, "simulator_provenance", lambda settings: deepcopy(provenance))
    monkeypatch.setattr(dataset, "render", fake_render)
    return cfg, calls, fake_render, provenance


def test_generation_extension_and_resume_equal_uninterrupted_run(generation_setup: tuple) -> None:
    cfg, calls, _, _ = generation_setup
    dataset.generate_dataset(cfg, "train", 3, 1)
    directory = dataset.split_directory(cfg, "train")
    original_csv = (directory / "params.csv").read_bytes()
    original_hashes = {p.name: dataset.file_sha256(p) for p in (directory / "pfm").glob("*.pfm")}
    summary = dataset.generate_dataset(cfg, "train", 8, 1)
    assert summary["n_skipped"] == 3 and summary["n_rendered_this_run"] == 5
    assert len(calls) == 8
    assert (directory / "params.csv").read_bytes().startswith(original_csv)
    assert all(
        dataset.file_sha256(directory / "pfm" / name) == digest
        for name, digest in original_hashes.items()
    )
    rerun = dataset.generate_dataset(cfg, "train", 8, 1)
    assert rerun["n_skipped"] == 8 and rerun["n_rendered_this_run"] == 0
    uninterrupted = deepcopy(cfg)
    uninterrupted["paths"]["data"] = "uninterrupted"
    dataset.generate_dataset(uninterrupted, "train", 8, 1)
    other = dataset.split_directory(uninterrupted, "train")
    assert (directory / "params.csv").read_bytes() == (other / "params.csv").read_bytes()
    for path in (directory / "pfm").glob("*.pfm"):
        assert path.read_bytes() == (other / "pfm" / path.name).read_bytes()


@pytest.mark.parametrize(
    "change", ["seed", "scene", "prior", "noise", "binary", "template", "parameters", "shrink"]
)
def test_resume_rejects_changed_identity_without_rendering(
    generation_setup: tuple, change: str
) -> None:
    cfg, calls, _, provenance = generation_setup
    dataset.generate_dataset(cfg, "train", 3, 1)
    n, seed = 3, 1
    directory = dataset.split_directory(cfg, "train")
    if change == "seed":
        seed = 3
    elif change == "scene":
        cfg["scene"]["supersample"] += 1
    elif change == "prior":
        cfg["prior"]["a_max"] = 0.9
    elif change == "noise":
        cfg["observation"]["sigma_n"] = 25.0
    elif change == "binary":
        provenance["binary_sha256"] = "different"
    elif change == "template":
        template = directory / "different.toml"
        template.write_text("different")
        cfg["paths"]["scene_template"] = str(template)
    elif change == "parameters":
        parameters = dataset.read_parameters(directory / "params.csv")
        parameters[0, 0] += 0.01
        (directory / "params.csv").write_text(dataset.parameter_csv(parameters))
    else:
        n = 2
    originals = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        dataset.generate_dataset(cfg, "train", n, seed)
    assert len(calls) == 3
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_interrupt_then_partial_preprocessing_and_resume(
    generation_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, calls, render, _ = generation_setup

    def interrupted(a: float, inclination: float, stem: Path, settings: Config) -> Path:
        if int(stem.name) == 2:
            raise KeyboardInterrupt
        return render(a, inclination, stem, settings)

    monkeypatch.setattr(dataset, "render", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dataset.generate_dataset(cfg, "train", 5, 1)
    metadata = dataset.preprocess_dataset(cfg, "train")
    assert metadata["n_requested"] == 5 and metadata["n_rendered"] == 2
    assert metadata["missing_indices"] == [2, 3, 4] and metadata["n_failed"] == 0
    arrays, _ = dataset.load_preprocessed(cfg, "train")
    np.testing.assert_array_equal(arrays["idx"], [0, 1])
    np.testing.assert_array_equal(arrays["theta"], sample_prior(5, 1, cfg)[:2, :2])
    monkeypatch.setattr(dataset, "render", render)
    summary = dataset.generate_dataset(cfg, "train", 5, 1)
    assert summary["n_rendered_this_run"] == 3 and summary["n_skipped"] == 2
    assert len(calls) == 5


def test_recover_after_pfm_rename_and_partial_ledger_write(
    generation_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, calls, _, _ = generation_setup
    append = dataset.append_event

    def interrupted(path: Path, event: dict) -> None:
        if event["idx"] == 1:
            with path.open("a") as stream:
                stream.write('{"idx":')
            raise KeyboardInterrupt
        append(path, event)

    monkeypatch.setattr(dataset, "append_event", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dataset.generate_dataset(cfg, "train", 4, 1)
    monkeypatch.setattr(dataset, "append_event", append)
    summary = dataset.generate_dataset(cfg, "train", 4, 1)
    assert summary["n_recovered"] == 1 and summary["n_skipped"] == 2
    assert summary["n_rendered_this_run"] == 2 and len(calls) == 4
    events = dataset.read_events(dataset.split_directory(cfg, "train") / "renders.jsonl")
    assert len(events) == 4 and events[1]["recovered_after_interruption"]


def test_retry_failure_reporting_and_partial_index_alignment(
    generation_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _, render, _ = generation_setup
    attempts = {}

    def unreliable(a: float, inclination: float, stem: Path, settings: Config) -> Path:
        index = int(stem.name)
        attempts[index] = attempts.get(index, 0) + 1
        if index == 1 or (index == 2 and attempts[index] == 1):
            raise RenderError("render failed", "specific stderr\nsecond line")
        return render(a, inclination, stem, settings)

    monkeypatch.setattr(dataset, "render", unreliable)
    summary = dataset.generate_dataset(cfg, "train", 4, 1)
    assert summary["n_failed"] == 1 and summary["n_rendered_this_run"] == 3
    assert attempts == {0: 1, 1: 2, 2: 2, 3: 1}
    directory = dataset.split_directory(cfg, "train")
    with (directory / "failures.csv").open() as stream:
        failures = list(csv.DictReader(stream))
    assert len(failures) == 1 and failures[0]["idx"] == "1"
    assert failures[0]["stderr"] == "specific stderr\nsecond line"
    metadata = dataset.preprocess_dataset(cfg, "train")
    arrays, _ = dataset.load_preprocessed(cfg, "train")
    assert metadata["n_failed"] == 1 and metadata["failed_indices"] == [1]
    np.testing.assert_array_equal(arrays["idx"], [0, 2, 3])
    np.testing.assert_array_equal(arrays["theta"], sample_prior(4, 1, cfg)[[0, 2, 3], :2])
    np.testing.assert_allclose(arrays["x"].sum(axis=(1, 2)), 4096, rtol=1e-7)
    monkeypatch.setattr(dataset, "render", render)
    assert dataset.generate_dataset(cfg, "train", 4, 1)["n_failed"] == 0
    with (directory / "failures.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []


def test_corrupt_or_missing_renders_are_repaired(generation_setup: tuple) -> None:
    cfg, calls, _, _ = generation_setup
    dataset.generate_dataset(cfg, "train", 4, 1)
    directory = dataset.split_directory(cfg, "train")
    paths = [directory / "pfm" / f"{index:06d}.pfm" for index in range(4)]
    originals = [p.read_bytes() for p in paths]
    paths[0].write_bytes(b"truncated")
    paths[1].write_bytes(originals[1][:-4] + np.asarray([9.5], dtype="<f4").tobytes())
    paths[2].unlink()
    metadata = dataset.preprocess_dataset(cfg, "train")
    assert set(metadata["invalid_indices"]) == {0, 1}
    assert metadata["missing_indices"] == [2]
    arrays, _ = dataset.load_preprocessed(cfg, "train")
    np.testing.assert_array_equal(arrays["idx"], [3])
    summary = dataset.generate_dataset(cfg, "train", 4, 1)
    assert summary["n_skipped"] == 1 and summary["n_rendered_this_run"] == 3
    assert len(calls) == 7
    assert [p.read_bytes() for p in paths] == originals


def test_interrupted_array_publication_is_detected_and_recoverable(
    generation_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _, _, _ = generation_setup
    dataset.generate_dataset(cfg, "train", 3, 1)
    dataset.preprocess_dataset(cfg, "train")
    dataset.generate_dataset(cfg, "train", 4, 1)
    replace = Path.replace

    def interrupted(path: Path, target: Path) -> Path:
        if path.name == "theta.npy" and path.parent.name.startswith(".preprocess-"):
            raise KeyboardInterrupt
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dataset.preprocess_dataset(cfg, "train")
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        dataset.load_preprocessed(cfg, "train")
    monkeypatch.setattr(Path, "replace", replace)
    dataset.preprocess_dataset(cfg, "train")
    arrays, metadata = dataset.load_preprocessed(cfg, "train")
    assert metadata["n_rendered"] == 4
    assert arrays["x"].shape == (4, 2, 2)
    np.testing.assert_array_equal(arrays["idx"], np.arange(4))


def test_train_and_test_cannot_share_prior_seed(generation_setup: tuple) -> None:
    cfg, _, _, _ = generation_setup
    dataset.generate_dataset(cfg, "train", 3, 1)
    with pytest.raises(ValueError, match="different prior seeds"):
        dataset.generate_dataset(cfg, "test", 3, 1)
    dataset.generate_dataset(cfg, "test", 3, 2)


def test_generation_lock_prevents_competing_writers(generation_setup: tuple) -> None:
    cfg, calls, _, _ = generation_setup
    with dataset.exclusive_lock(project_path(cfg, "data") / ".generation.lock"):
        with pytest.raises(RuntimeError, match="another process"):
            dataset.generate_dataset(cfg, "train", 3, 1)
    assert not calls


def test_preprocessing_rejects_changed_observation_model(generation_setup: tuple) -> None:
    cfg, _, _, _ = generation_setup
    dataset.generate_dataset(cfg, "train", 3, 1)
    cfg["observation"]["sigma_psf"] = 2.0
    with pytest.raises(ValueError, match="configuration differs"):
        dataset.preprocess_dataset(cfg, "train")


def test_preprocessing_keeps_snapshot_when_parameter_table_grows(
    generation_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _, _, _ = generation_setup
    dataset.generate_dataset(cfg, "train", 3, 1)
    directory = dataset.split_directory(cfg, "train")
    initial = (directory / "params.csv").read_bytes()
    read_events = dataset.read_events

    def extend_then_read(path: Path, *, repair: bool = False) -> dict:
        if not repair:
            dataset.generate_dataset(cfg, "train", 5, 1)
        return read_events(path, repair=repair)

    monkeypatch.setattr(dataset, "read_events", extend_then_read)
    metadata = dataset.preprocess_dataset(cfg, "train")
    assert metadata["n_requested"] == 3 and metadata["n_rendered"] == 3
    assert metadata["params_sha256"] == hashlib.sha256(initial).hexdigest()
    assert len(dataset.read_parameters(directory / "params.csv")) == 5
    arrays, _ = dataset.load_preprocessed(cfg, "train")
    np.testing.assert_array_equal(arrays["idx"], [0, 1, 2])


def test_empty_partial_dataset_has_correct_array_shapes(generation_setup: tuple) -> None:
    cfg, calls, _, _ = generation_setup
    dataset.prepare_generation(cfg, "train", 3, 1)
    metadata = dataset.preprocess_dataset(cfg, "train")
    arrays, _ = dataset.load_preprocessed(cfg, "train")
    assert metadata["n_rendered"] == 0 and metadata["n_failed"] == 0
    assert metadata["missing_indices"] == [0, 1, 2]
    assert arrays["x"].shape == (0, 2, 2)
    assert arrays["theta"].shape == (0, 2) and arrays["idx"].shape == (0,)
    assert not calls


def test_generator_cli_fails_when_persistent_failures_remain(
    generation_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _, _, _ = generation_setup
    path = Path(__file__).resolve().parents[1] / "scripts/01_generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr("sys.argv", [str(path), "--split", "train", "--n", "2", "--seed", "1"])

    def fail(*args: object) -> None:
        raise RenderError("failure", "stderr")

    monkeypatch.setattr(dataset, "render", fail)
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1
    directory = dataset.split_directory(cfg, "train")
    assert json.loads((directory / "generation_summary.json").read_text())["n_failed"] == 2


def test_unproven_existing_render_is_not_adopted(generation_setup: tuple) -> None:
    cfg, calls, _, _ = generation_setup
    directory = dataset.split_directory(cfg, "train") / "pfm"
    directory.mkdir(parents=True)
    (directory / "000000.pfm").write_bytes(b"unproven")
    with pytest.raises(ValueError, match="provenance"):
        dataset.generate_dataset(cfg, "train", 3, 1)
    assert not calls
