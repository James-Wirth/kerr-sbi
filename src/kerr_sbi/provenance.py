import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from kerr_sbi.config import Config
from kerr_sbi.simulator import nullgeo_binary


def _repository_metadata(repository: Path) -> tuple[str, str, bool]:
    manifest = tomllib.loads((repository / "crates/nullgeo-cli/Cargo.toml").read_text())
    version = manifest["package"]["version"]
    if isinstance(version, dict) and version.get("workspace"):
        workspace = tomllib.loads((repository / "Cargo.toml").read_text())
        version = workspace["workspace"]["package"]["version"]
    command = ["git", "--no-optional-locks", "-C", str(repository)]
    revision = subprocess.run(
        [*command, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    status = subprocess.run(
        [*command, "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    return version, revision, bool(status)


def _cargo_metadata(executable: Path) -> tuple[str, str]:
    manifest = executable.parent.parent / ".crates.toml"
    if not manifest.is_file():
        raise ValueError(
            "Cannot identify this nullgeo build; set NULLGEO_REPO to its source checkout"
        )
    installs = tomllib.loads(manifest.read_text()).get("v1", {})
    packages = [
        package
        for package, binaries in installs.items()
        if package.startswith("nullgeo-cli ") and executable.name in binaries
    ]
    if (
        len(packages) != 1
        or "(registry+https://github.com/rust-lang/crates.io-index)" not in packages[0]
    ):
        raise ValueError("Cannot identify a crates.io nullgeo build; set NULLGEO_REPO")
    version = packages[0].split()[1]
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).expanduser()
    metadata_files = list(
        (cargo_home / "registry/src").glob(f"*/nullgeo-cli-{version}/.cargo_vcs_info.json")
    )
    revisions = {json.loads(path.read_text())["git"]["sha1"] for path in metadata_files}
    if len(revisions) != 1:
        raise ValueError("Nullgeo source metadata is missing or inconsistent; set NULLGEO_REPO")
    return version, revisions.pop()


def simulator_provenance(cfg: Config) -> dict[str, str | bool]:
    binary = shutil.which(nullgeo_binary())
    if binary is None:
        raise FileNotFoundError(f"nullgeo executable not found: {nullgeo_binary()}")
    executable = Path(binary).resolve()
    repository = os.environ.get("NULLGEO_REPO")
    if repository:
        version, revision, dirty = _repository_metadata(Path(repository).expanduser().resolve())
        evidence = "NULLGEO_REPO; source checkout is not proof of which source built the executable"
    else:
        version, revision = _cargo_metadata(executable)
        dirty = False
        evidence = "Cargo installation and cached registry source metadata"
    if version != cfg["simulator"]["version"]:
        raise ValueError(f"Expected nullgeo {cfg['simulator']['version']}, found {version}")
    return {
        "package": "nullgeo-cli",
        "version": version,
        "source_commit": revision,
        "source_commit_evidence": evidence,
        "source_worktree_dirty": dirty,
        "binary": str(executable),
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
