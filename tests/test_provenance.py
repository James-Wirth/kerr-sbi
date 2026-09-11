import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import tomli_w

from kerr_sbi.config import Config
from kerr_sbi.provenance import simulator_provenance


@pytest.fixture
def installed_nullgeo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "install/bin/nullgeo"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"simulator executable")
    (tmp_path / "install/.crates.toml").write_text(
        tomli_w.dumps(
            {
                "v1": {
                    "nullgeo-cli 0.2.0 (registry+https://github.com/rust-lang/crates.io-index)": [
                        "nullgeo"
                    ]
                }
            }
        )
    )
    metadata = tmp_path / "cargo/registry/src/registry/nullgeo-cli-0.2.0/.cargo_vcs_info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"git": {"sha1": "a" * 40}}))
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "cargo"))
    monkeypatch.delenv("NULLGEO_REPO", raising=False)
    monkeypatch.setattr("kerr_sbi.provenance.shutil.which", lambda _: str(executable))
    return executable


def test_cargo_provenance_needs_no_saved_results(installed_nullgeo: Path, cfg: Config) -> None:
    cfg["paths"] = {}
    result = simulator_provenance(cfg)
    assert result["version"] == "0.2.0"
    assert result["source_commit"] == "a" * 40
    assert result["source_worktree_dirty"] is False
    assert result["binary_sha256"] == hashlib.sha256(installed_nullgeo.read_bytes()).hexdigest()


def test_rejects_unexpected_version(installed_nullgeo: Path, cfg: Config) -> None:
    cfg["simulator"]["version"] = "0.1.1"
    with pytest.raises(ValueError, match="Expected nullgeo 0.1.1, found 0.2.0"):
        simulator_provenance(cfg)


def test_missing_source_metadata_has_actionable_error(
    installed_nullgeo: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="source metadata.*NULLGEO_REPO"):
        simulator_provenance(cfg)


def test_conflicting_source_metadata_is_rejected(
    installed_nullgeo: Path, cfg: Config, tmp_path: Path
) -> None:
    second = tmp_path / "cargo/registry/src/second/nullgeo-cli-0.2.0/.cargo_vcs_info.json"
    second.parent.mkdir(parents=True)
    second.write_text(json.dumps({"git": {"sha1": "b" * 40}}))
    with pytest.raises(ValueError, match="inconsistent"):
        simulator_provenance(cfg)


@pytest.mark.parametrize("workspace_version", [False, True])
@pytest.mark.parametrize("dirty", [False, True])
def test_repository_provenance_records_version_commit_and_local_changes(
    installed_nullgeo: Path,
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_version: bool,
    dirty: bool,
) -> None:
    repository = tmp_path / "source"
    manifest = repository / "crates/nullgeo-cli/Cargo.toml"
    manifest.parent.mkdir(parents=True)
    version = {"workspace": True} if workspace_version else "0.2.0"
    manifest.write_text(tomli_w.dumps({"package": {"version": version}}))
    (repository / "Cargo.toml").write_text(
        tomli_w.dumps({"workspace": {"package": {"version": "0.2.0"}}})
    )
    monkeypatch.setenv("NULLGEO_REPO", str(repository))

    def git_read(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["git", "--no-optional-locks", "-C", str(repository)]
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 10
        output = "b" * 40 if command[4] == "rev-parse" else " M source.rs" if dirty else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("kerr_sbi.provenance.subprocess.run", git_read)
    result = simulator_provenance(cfg)
    assert result["source_commit"] == "b" * 40
    assert result["version"] == "0.2.0"
    assert result["source_worktree_dirty"] is dirty
