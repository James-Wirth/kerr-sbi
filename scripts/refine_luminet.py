import argparse
import json
import shutil
import signal
import subprocess
import time
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from replicate_luminet import IDENTITY, ROOT, compare, write_json

from kerr_sbi.luminet import (
    corrected_redshift,
    impact_coordinates,
    reconstruct_rings,
    redshift,
)
from kerr_sbi.transfer import GeometryCache, geometry_key, sha256


def render_plan(parent: dict) -> list:
    base = next(row["scene"] for row in parent["renders"] if row["name"] == "R4000_n7")
    rows = []
    for name, jitter, tolerance in [
        ("R4000_n9", False, parent["settings"]["tolerance"]),
        ("R4000_jitter9", True, parent["settings"]["tolerance"]),
        ("R4000_tight9", False, parent["settings"]["tight_tolerance"]),
    ]:
        scene = deepcopy(base)
        scene["camera"].update(supersample=9, supersample_max=9, jitter=jitter)
        scene["integrator"]["tol"] = tolerance
        rows.append(dict(name=name, scene=scene))
    return rows


def require_cached(cache: GeometryCache, scene: dict) -> None:
    path = cache.root / geometry_key(scene, cache.identity) / "cache.json"
    if not path.is_file():
        raise FileNotFoundError(f"required cache absent; rerender forbidden: {path}")


def freeze(destination: Path) -> None:
    if (destination / "protocol.json").exists():
        raise FileExistsError("already frozen")
    config = ROOT / "configs/luminet_refinement.toml"
    settings = tomllib.loads(config.read_text())
    if settings["sampling"] != [5, 7, 9] or settings["render_budget"] != 3:
        raise ValueError("not the authorized continuation")
    old = ROOT / "runs" / settings["parent"]
    manifest = json.loads((old / "artifact_manifest.json").read_text())
    for name, digest in manifest.items():
        if sha256(old / name) != digest:
            raise ValueError(f"parent artifact changed: {name}")
    parent = json.loads((old / "protocol.json").read_text())
    if sha256(Path(parent["binary"]["binary_path"])) != parent["binary"]["binary_sha256"]:
        raise ValueError("binary identity mismatch")
    if json.loads(IDENTITY.read_text()) != parent["binary"]:
        raise ValueError("adjacent identity mismatch")
    protected = dict(parent["protected"])
    for path in old.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            protected[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in protected.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"protected artifact changed: {name}")
    inputs = [
        Path(__file__),
        config,
        ROOT / "scripts/replicate_luminet.py",
        ROOT / "src/kerr_sbi/luminet.py",
        ROOT / "src/kerr_sbi/transfer.py",
        ROOT / "tests/test_luminet_refinement.py",
        ROOT / "uv.lock",
        destination / "PROTOCOL.md",
        old / "artifact_manifest.json",
    ]
    snapshots = destination / "inputs"
    snapshots.mkdir()
    for path in inputs:
        shutil.copyfile(path, snapshots / path.name)
    shutil.copyfile(old / "PROTOCOL.md", snapshots / "original_PROTOCOL.md")
    protocol = dict(
        frozen_utc=datetime.now(UTC).isoformat(),
        settings=settings,
        parent=parent,
        parent_path=str(old),
        renders=render_plan(parent),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree=subprocess.check_output(["git", "status", "--short"], text=True),
        inputs={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        protected=protected,
        commands=[
            f".venv/bin/python scripts/refine_luminet.py --run {destination.name} --freeze",
            f".venv/bin/python scripts/refine_luminet.py --run {destination.name}",
        ],
    )
    write_json(destination / "protocol.json", protocol)
    (destination / "protocol.sha256").write_text(sha256(destination / "protocol.json") + "\n")
    print("Frozen", protocol["frozen_utc"], flush=True)


def execute(destination: Path) -> None:
    if (destination / "execution.json").exists():
        raise FileExistsError("execution already attempted")
    start = time.perf_counter()
    protocol = json.loads((destination / "protocol.json").read_text())
    if (
        sha256(destination / "protocol.json")
        != (destination / "protocol.sha256").read_text().strip()
    ):
        raise ValueError("protocol changed")
    for name, digest in {**protocol["inputs"], **protocol["protected"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    cfg, parent = protocol["settings"], protocol["parent"]
    settings, old = parent["settings"], Path(protocol["parent_path"])

    def deadline(signum: int, frame: object) -> None:
        raise TimeoutError("continuation deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, cfg["max_seconds"] - (time.perf_counter() - start))
    write_json(
        destination / "execution.json",
        dict(started_utc=datetime.now(UTC).isoformat(), command=protocol["commands"][1]),
    )
    saved = np.load(old / "reference.npz")
    reference = dict(
        b=saved["4000_b"], z=saved["4000_z"], valid=np.ones_like(saved["4000_b"], dtype=bool)
    )
    beta, radii, distance = saved["beta"], saved["radii"], settings["principal_distance"]
    binary = Path(parent["binary"]["binary_path"])
    old_cache = GeometryCache(old / "geometry", binary, parent["binary"])
    cache = GeometryCache(destination / "geometry", binary, parent["binary"])
    reused = [row for row in parent["renders"] if row["name"] in ["R4000_n5", "R4000_n7"]]
    products, transfers, metrics, attempts = {}, {}, {}, []
    for row in reused + protocol["renders"]:
        name, scene = row["name"], row["scene"]
        is_reused = name in ["R4000_n5", "R4000_n7"]
        if is_reused:
            require_cached(old_cache, scene)
            transfer = old_cache.get(scene)
        else:
            if len(attempts) >= cfg["render_budget"]:
                raise RuntimeError("render budget exhausted")
            size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
            if size + 50_000_000 > cfg["storage_budget_bytes"]:
                raise RuntimeError("storage ceiling")
            attempts.append(dict(name=name, state="started"))
            write_json(destination / "renders.json", attempts)
            print("Render", name, "elapsed", time.perf_counter() - start, flush=True)
            transfer = cache.get(
                scene,
                timeout=min(
                    cfg["render_timeout"], cfg["max_seconds"] - (time.perf_counter() - start)
                ),
            )
            record = json.loads(
                (cache.root / geometry_key(scene, parent["binary"]) / "cache.json").read_text()
            )
            attempts[-1].update(
                state="completed",
                seconds=record["seconds"],
                command=record["command"],
                key=record["key"],
            )
            write_json(destination / "renders.json", attempts)
        transfers[name] = transfer
        product = reconstruct_rings(
            transfer.samples,
            distance,
            beta,
            radii,
            reference["b"],
            settings["neighbours"],
            settings["stencil_distance"],
            settings["stencil_radius_span"],
            settings["max_offset"],
        )
        if is_reused:
            original = np.load(old / f"{name}_analysis.npz")
            for key, value in product.items():
                np.testing.assert_array_equal(value, original[key])
        products[name] = product
        s = transfer.samples
        hits = s["has_intersection"].astype(bool)
        available = hits & s["g_valid"].astype(bool) & s["g_available"].astype(bool)
        xy = impact_coordinates(s["screen_u"], s["screen_v"], distance)
        residual = np.full(len(s), np.nan)
        residual[available] = corrected_redshift(s["g"][available], distance) - redshift(
            s["radius"][available], xy[available, 0], settings["inclination"]
        )
        conditional = dict(
            p99=float(np.quantile(np.abs(residual[available]), 0.99)),
            max=float(np.max(np.abs(residual[available]))),
        )
        conditional["passed"] = all(conditional[k] <= v for k, v in settings["conditional"].items())
        diagnostics = transfer.diagnostics()
        clean = all(
            diagnostics[k] == 0
            for k in [
                "unfinished",
                "invalid_radius",
                "invalid_g",
                "invalid_radiance",
                "missing_g_at_hit",
            ]
        )
        metrics[name] = dict(
            matched=compare(product, reference, settings["matched"]),
            conditional=conditional,
            diagnostics=diagnostics,
            clean=clean,
            reused=is_reused,
            reused_arrays_exact=True if is_reused else None,
        )
        np.savez_compressed(
            destination / f"{name}_analysis.npz",
            **product,
            conditional_residual=residual,
            hit_mask=hits,
            available_mask=available,
        )
        write_json(destination / "metrics_partial.json", metrics)
        print(name, metrics[name], flush=True)
    pairs = [
        ("R4000_n5", "R4000_n7"),
        ("R4000_n7", "R4000_n9"),
        ("R4000_n9", "R4000_jitter9"),
        ("R4000_n9", "R4000_tight9"),
    ]
    refinement = {
        a + "_to_" + b: compare(products[a], products[b], settings["refinement"]) for a, b in pairs
    }
    a, b = transfers["R4000_n9"].samples, transfers["R4000_tight9"].samples
    for key in ["screen_u", "screen_v", "weight", "pixel_index", "sample_index"]:
        np.testing.assert_array_equal(a[key], b[key])
    shared = a["has_intersection"].astype(bool) & b["has_intersection"].astype(bool)
    tolerance = dict(
        hit_changes=int(np.count_nonzero(a["has_intersection"] != b["has_intersection"])),
        outcome_changes=int(np.count_nonzero(a["outcome"] != b["outcome"])),
        radius_max=float(np.max(np.abs(a["radius"][shared] - b["radius"][shared]))),
        g_max=float(np.max(np.abs(a["g"][shared] - b["g"][shared]))),
    )
    tolerance["passed"] = tolerance["hit_changes"] == tolerance["outcome_changes"] == 0 and all(
        tolerance[k] <= v for k, v in settings["tolerance_check"].items()
    )
    old_summary = json.loads((old / "summary.json").read_text())
    summary = dict(
        metrics=metrics,
        refinement=refinement,
        tolerance=tolerance,
        finite_distance=old_summary["finite_distance"],
        reference_checks=old_summary["reference_checks"],
        distant_approximation_pass=old_summary["distant_approximation_pass"],
        matched_pass=all(metrics[n]["matched"]["passed"] for n in ["R4000_n9", "R4000_jitter9"]),
        refinement_pass=all(r["passed"] for r in refinement.values()),
        conditional_pass=all(r["conditional"]["passed"] for r in metrics.values()),
        clean=all(r["clean"] for r in metrics.values()),
        render_count=len(attempts),
        render_seconds=sum(r["seconds"] for r in attempts),
        historical_artifacts_unchanged=all(
            sha256(ROOT / name) == digest for name, digest in protocol["protected"].items()
        ),
        driver_seconds=time.perf_counter() - start,
        artifact_bytes=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
    )
    summary["reproduced"] = all(
        summary[k]
        for k in [
            "matched_pass",
            "refinement_pass",
            "conditional_pass",
            "clean",
            "historical_artifacts_unchanged",
        ]
    )
    summary["reproduced"] &= tolerance["passed"] and summary["distant_approximation_pass"]
    write_json(destination / "summary.json", summary)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print("Completed", {k: v for k, v in summary.items() if not isinstance(v, dict)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("run must be one directory name")
    destination = ROOT / "runs" / args.run
    if args.freeze:
        freeze(destination)
    else:
        try:
            execute(destination)
        except Exception as exc:
            write_json(
                destination / "failure.json", dict(type=type(exc).__name__, message=str(exc))
            )
            raise


if __name__ == "__main__":
    main()
