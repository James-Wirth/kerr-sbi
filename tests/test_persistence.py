from pathlib import Path

import pytest

from kerr_sbi.persistence import atomic_text, exclusive_lock, file_sha256, write_json


def test_atomic_text_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "nested/value.txt"
    atomic_text(path, "old")
    atomic_text(path, "abc")
    assert path.read_bytes() == b"abc"
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert list(path.parent.iterdir()) == [path]


def test_failed_atomic_write_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "value.txt"
    path.write_text("old")

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_text(path, "new")
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_json_format_and_nonfinite_rejection(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    write_json(path, {"value": 1})
    expected = b'{\n  "value": 1\n}\n'
    assert path.read_bytes() == expected
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert path.read_bytes() == expected


def test_exclusive_lock_releases_after_error(tmp_path: Path) -> None:
    path = tmp_path / "nested/lock"
    with pytest.raises(ValueError, match="operation failed"), exclusive_lock(path):
        with pytest.raises(RuntimeError, match="another process"):
            with exclusive_lock(path):
                pytest.fail("contended lock acquired")
        raise ValueError("operation failed")
    with exclusive_lock(path):
        assert path.exists()
