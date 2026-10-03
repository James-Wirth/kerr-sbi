import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from kerr_sbi.transfer import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    repo, destination = args.repo.resolve(), args.destination.resolve()

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()

    if git("rev-parse", "HEAD") != args.commit or git("status", "--porcelain"):
        raise ValueError("renderer must be clean at requested commit")
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "source.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", f"--output={archive}", args.commit],
        check=True,
    )
    command = [
        "cargo",
        "build",
        "--offline",
        "--locked",
        "--release",
        "--manifest-path",
        str(repo / "Cargo.toml"),
        "-p",
        "nullgeo-cli",
        "--target-dir",
        str(destination / "target"),
    ]
    start = time.perf_counter()
    with (destination / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = destination / "target/release/nullgeo"
    record = {
        "binary_path": str(binary),
        "binary_sha256": sha256(binary),
        "source_commit": args.commit,
        "source_state": "clean",
        "source_archive_sha256": sha256(archive),
        "cargo_lock_sha256": sha256(repo / "Cargo.lock"),
        "command": command,
        "seconds": time.perf_counter() - start,
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True),
        "platform": platform.platform(),
        "features": ["default", "parallel"],
        "profile": "release",
        "environment": {k: os.environ.get(k) for k in ("RUSTFLAGS", "CARGO_BUILD_TARGET")},
    }
    if git("rev-parse", "HEAD") != args.commit or git("status", "--porcelain"):
        raise ValueError("renderer changed during build")
    (destination / "identity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
