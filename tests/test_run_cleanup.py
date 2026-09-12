import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MONITOR = Path(__file__).resolve().parents[1] / "tools/training_monitor"
sys.path.insert(0, str(MONITOR))
spec = importlib.util.spec_from_file_location("monitor_server", MONITOR / "server.py")
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
sys.path.pop(0)
Records = server_module.Records
lock = sys.modules["lifecycle"].lock


@pytest.fixture
def reader(tmp_path: Path) -> Records:
    data = tmp_path / "data/datasets/reusable/train"
    data.mkdir(parents=True)
    (data / "x.npy").write_bytes(b"shared simulation inputs")
    directory = tmp_path / "runs/example"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps({"config": {"paths": {"data": "data/datasets/reusable"}}})
    )
    (directory / "weights.bin").write_bytes(b"weights and outputs")
    return Records(tmp_path)


def test_trash_restore_and_permanent_delete_preserve_shared_data(reader: Records) -> None:
    directory = reader.runs / "example"
    before = {
        p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }
    shared = reader.data / "datasets/reusable/train/x.npy"
    shared_before = (shared.read_bytes(), shared.stat().st_mtime_ns)
    entry = reader.trash.trash("example")
    assert not directory.exists()
    assert reader.overview()["runs"] == []
    assert reader.trash.entries()[0]["id"] == entry["id"]
    reader.trash.restore(entry["id"])
    assert before == {
        p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }
    entry = reader.trash.trash("example")
    assert reader.trash.purge([entry["id"]]) == {"deleted": 1}
    assert reader.trash.entries() == []
    assert (shared.read_bytes(), shared.stat().st_mtime_ns) == shared_before


@pytest.mark.parametrize("suffix", [".lock", ".job.lock"])
def test_active_training_lock_prevents_cleanup(reader: Records, suffix: str) -> None:
    with lock(reader.runs / f".example{suffix}"):
        assert reader.run("example")["active"]
        with pytest.raises(RuntimeError, match="busy"):
            reader.trash.trash("example")
    assert not reader.run("example")["active"]
    assert (reader.runs / "example/run.json").exists()


@pytest.mark.parametrize(
    "identifier", ["", ".", "..", "../data", "example/../example", ".trash", "/example"]
)
def test_cleanup_rejects_non_run_paths(reader: Records, identifier: str) -> None:
    with pytest.raises(ValueError):
        reader.trash.trash(identifier)
    assert (reader.runs / "example/weights.bin").exists()


def test_cleanup_rejects_symlinks_and_nested_simulations(reader: Records) -> None:
    (reader.runs / "alias").symlink_to(reader.runs / "example", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        reader.trash.trash("alias")
    nested = reader.runs / "example/data/train"
    nested.mkdir(parents=True)
    (nested / "generation.json").write_text("{}")
    with pytest.raises(ValueError, match="simulation data"):
        reader.trash.trash("example")


def test_restore_collision_and_invalid_purge_do_not_remove_runs(reader: Records) -> None:
    entry = reader.trash.trash("example")
    (reader.runs / "example").mkdir()
    with pytest.raises(FileExistsError):
        reader.trash.restore(entry["id"])
    with pytest.raises(ValueError):
        reader.trash.purge([entry["id"], "example"])
    assert len(reader.trash.entries()) == 1
    with pytest.raises(ValueError):
        reader.trash.purge([entry["id"], entry["id"]])


def test_trash_symlink_cannot_redirect_permanent_deletion(reader: Records) -> None:
    entry = reader.trash.trash("example")
    envelope = reader.trash.directory / entry["id"]
    (envelope / "run").rename(envelope / "saved")
    (envelope / "run").symlink_to(reader.data, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        reader.trash.purge([entry["id"]])
    assert (reader.data / "datasets/reusable/train/x.npy").exists()


def request(
    handler: type, method: str, path: str, payload: dict | None = None, **headers: str
) -> tuple[int, bytes]:
    body = json.dumps(payload).encode() if payload is not None else b""
    fields = {"Host": "127.0.0.1:8765", "Content-Length": str(len(body)), **headers}
    head = (
        f"{method} {path} HTTP/1.0\r\n"
        + "".join(f"{key}: {value}\r\n" for key, value in fields.items())
        + "\r\n"
    )
    output = bytearray()
    connection = SimpleNamespace(
        makefile=lambda *args: io.BytesIO(head.encode() + body), sendall=output.extend
    )
    handler(connection, ("127.0.0.1", 10000), SimpleNamespace(server_port=8765))
    status, response = bytes(output).split(b"\r\n\r\n", 1)
    return int(status.split()[1]), response


def test_http_cleanup_requires_origin_token_and_explicit_purge(reader: Records) -> None:
    handler = server_module.handler_for(reader)
    status, body = request(handler, "GET", "/api/overview")
    assert status == 200
    token = json.loads(body)["mutation_token"]
    headers = {
        "Origin": "http://127.0.0.1:8765",
        "X-Monitor-Token": token,
        "Content-Type": "application/json",
    }
    assert request(handler, "POST", "/api/runs/trash", {"id": "example"})[0] == 403
    assert (
        request(
            handler,
            "POST",
            "/api/runs/trash",
            {"id": "example"},
            **{**headers, "Origin": "https://foreign.example"},
        )[0]
        == 403
    )
    assert (
        request(
            handler,
            "POST",
            "/api/runs/trash",
            {"id": "example"},
            **{**headers, "X-Monitor-Token": "wrong"},
        )[0]
        == 403
    )
    status, body = request(handler, "POST", "/api/runs/trash", {"id": "example"}, **headers)
    assert status == 200
    identifier = json.loads(body)["id"]
    assert request(handler, "POST", "/api/trash/purge", {"ids": [identifier]}, **headers)[0] == 409
    assert request(handler, "POST", "/api/trash/restore", {"id": identifier}, **headers)[0] == 200
    assert request(handler, "GET", "/api/figure?id=example&name=../../run")[0] == 400
    assert request(handler, "GET", "/api/overview", **{"Host": "foreign.example"})[0] == 403
