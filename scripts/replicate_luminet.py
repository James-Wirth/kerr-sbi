import argparse
import json
import shutil
import signal
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kerr_sbi.luminet import (
    corrected_redshift,
    impact_coordinates,
    reconstruct_rings,
    redshift,
    residual_metrics,
    ring_reference,
)
from kerr_sbi.transfer import GeometryCache, geometry_key, sha256

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / ".tools/nullgeo-transfer-c2a46a6-20261003/identity.json"


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scene(settings: dict, distance: float, sampling: int, jitter: bool, tol: float) -> dict:
    angle = np.deg2rad(settings["inclination"])
    return {
        "metric": {"kind": "kerr", "mass": 1.0, "spin": 0.0},
        "camera": {
            "position": [-distance * np.sin(angle), 0.0, distance * np.cos(angle)],
            "fov_deg": float(2 * np.rad2deg(np.arctan(settings["screen_half_width"] / distance))),
            "width": settings["width"],
            "height": settings["width"],
            "supersample": sampling,
            "supersample_max": sampling,
            "jitter": jitter,
        },
        "disk": {
            "model": "stylized",
            "r_in": 0.0,
            "r_out": settings["r_out"],
            "emissivity_index": 2.0,
            "g_power": 3.0,
        },
        "sky": {"uniform": [0.0, 0.0, 0.0]},
        "integrator": {"tol": tol, "max_steps": 20000},
    }


def plan(settings: dict) -> list:
    distance = settings["principal_distance"]
    normal, tight = settings["tolerance"], settings["tight_tolerance"]
    rows = [(f"R4000_n{n}", distance, n, False, normal) for n in settings["sampling"]]
    rows += [
        ("R4000_jitter7", distance, 7, True, normal),
        ("R4000_tight7", distance, 7, False, tight),
        ("R2000_n7", 2000.0, 7, False, normal),
        ("R1000_n7", 1000.0, 7, False, normal),
    ]
    return [
        dict(name=name, distance=r, scene=scene(settings, r, n, jitter, tol))
        for name, r, n, jitter, tol in rows
    ]


def freeze(destination: Path) -> None:
    if (destination / "protocol.json").exists():
        raise FileExistsError("protocol already frozen")
    settings = tomllib.loads((ROOT / "configs/luminet.toml").read_text())
    identity = json.loads(IDENTITY.read_text())
    for path, key in [
        (Path(identity["binary_path"]), "binary_sha256"),
        (IDENTITY.parent / "source.tar", "source_archive_sha256"),
    ]:
        if sha256(path) != identity[key]:
            raise ValueError("retained identity mismatch")
    protected = {}
    for directory in [
        ROOT / ".tools",
        *[p for p in (ROOT / "runs").iterdir() if p.is_dir() and p != destination],
    ]:
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                protected[str(path.relative_to(ROOT))] = sha256(path)
    inputs = [
        ROOT / "src/kerr_sbi/luminet.py",
        ROOT / "src/kerr_sbi/transfer.py",
        ROOT / "scripts/replicate_luminet.py",
        ROOT / "configs/luminet.toml",
        ROOT / "tests/test_luminet.py",
        ROOT / "uv.lock",
        IDENTITY,
        destination / "PROTOCOL.md",
        destination / "paper.pdf",
    ]
    snapshot = destination / "inputs"
    snapshot.mkdir()
    for path in inputs:
        shutil.copyfile(path, snapshot / path.name)
    protocol = dict(
        frozen_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        working_tree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        renderer_working_tree=subprocess.check_output(
            ["git", "-C", "/Users/jameswirth/RustProjects/grrt", "status", "--short"], text=True
        ),
        binary=identity,
        settings=settings,
        renders=plan(settings),
        inputs={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        protected=protected,
        paper_url="https://articles.adsabs.harvard.edu/pdf/1979A%26A....75..228L",
        retrieval_url="https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?"
        "1979A%26A....75..228L&defaultprint=YES&filetype=.pdf",
        commands=[
            ".venv/bin/python scripts/replicate_luminet.py --run " + destination.name + " --freeze",
            ".venv/bin/python scripts/replicate_luminet.py --run " + destination.name,
        ],
    )
    write_json(destination / "protocol.json", protocol)
    (destination / "protocol.sha256").write_text(sha256(destination / "protocol.json") + "\n")
    print("Frozen protocol", protocol["frozen_utc"], flush=True)


def compare(a: dict, b: dict, gates: dict) -> dict:
    rows = []
    for j in range(a["b"].shape[0]):
        metrics = {key: residual_metrics(a[key][j] - b[key][j]) for key in ["b", "z"]}
        valid = bool(np.all(a["valid"][j]) and np.all(b["valid"][j]))
        passed = valid and all(
            metrics[key][stat] is not None and metrics[key][stat] <= gates[f"{key}_{stat}"]
            for key in ["b", "z"]
            for stat in ["rms", "max"]
        )
        rows.append(dict(**metrics, valid=valid, passed=passed))
    return dict(rings=rows, passed=all(row["passed"] for row in rows))


def execute(destination: Path) -> None:
    if (destination / "execution.json").exists():
        raise FileExistsError("execution already attempted; historical runs are immutable")
    start = time.perf_counter()
    protocol = json.loads((destination / "protocol.json").read_text())
    if (
        sha256(destination / "protocol.json")
        != (destination / "protocol.sha256").read_text().strip()
    ):
        raise ValueError("frozen protocol changed")
    for name, digest in protocol["inputs"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    settings = protocol["settings"]

    def deadline(signum: int, frame: object) -> None:
        raise TimeoutError("study computation ceiling reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, settings["max_seconds"])
    write_json(
        destination / "execution.json",
        dict(started_utc=datetime.now(UTC).isoformat(), command=protocol["commands"][1]),
    )
    radii = np.asarray(settings["radii"])
    beta = np.linspace(0, 2 * np.pi, settings["angle_nodes"], endpoint=False)
    references, reference_checks, archives = {}, {}, dict(beta=beta, radii=radii)
    for distance in [*settings["distances"], np.inf]:
        label = str(int(distance)) if np.isfinite(distance) else "infinity"
        b = ring_reference(
            radii, beta, settings["inclination"], distance, settings["reference_tolerance"]
        )
        tight = ring_reference(
            radii, beta, settings["inclination"], distance, settings["reference_tight_tolerance"]
        )
        z = redshift(radii[:, None], b * np.cos(beta), settings["inclination"])
        tight_z = redshift(radii[:, None], tight * np.cos(beta), settings["inclination"])
        reference_checks[label] = dict(
            b_max=float(np.max(np.abs(b - tight))), z_max=float(np.max(np.abs(z - tight_z)))
        )
        if any(reference_checks[label][k] > v for k, v in settings["reference_check"].items()):
            raise ValueError("independent reference tightening failed")
        references[label] = dict(b=b, z=z, valid=np.ones_like(b, dtype=bool))
        archives.update({label + "_b": b, label + "_z": z, label + "_tight_b": tight})
        print("Reference", label, reference_checks[label], flush=True)
    np.savez_compressed(destination / "reference.npz", **archives)
    write_json(destination / "reference_checks.json", reference_checks)
    cache = GeometryCache(
        destination / "geometry", Path(protocol["binary"]["binary_path"]), protocol["binary"]
    )
    products, metrics, renders = {}, {}, []
    transfers = {}
    for entry in protocol["renders"]:
        if len(renders) >= settings["render_budget"]:
            raise RuntimeError("render budget exhausted")
        if (
            sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
            > settings["storage_budget_bytes"] - 100_000_000
        ):
            raise RuntimeError("storage guard")
        name, distance = entry["name"], entry["distance"]
        print("Render", name, "elapsed", time.perf_counter() - start, flush=True)
        transfer = cache.get(
            entry["scene"],
            timeout=min(
                settings["render_timeout"], settings["max_seconds"] - (time.perf_counter() - start)
            ),
        )
        transfers[name] = transfer
        metadata = json.loads(
            (
                cache.root / geometry_key(entry["scene"], protocol["binary"]) / "cache.json"
            ).read_text()
        )
        renders.append(
            dict(
                name=name,
                seconds=metadata["seconds"],
                command=metadata["command"],
                key=metadata["key"],
            )
        )
        write_json(destination / "renders.json", renders)
        s = transfer.samples
        hits = s["has_intersection"].astype(bool)
        available = hits & s["g_valid"].astype(bool) & s["g_available"].astype(bool)
        xy = impact_coordinates(s["screen_u"], s["screen_v"], distance)
        conditional = np.full(len(s), np.nan)
        conditional[available] = corrected_redshift(s["g"][available], distance) - redshift(
            s["radius"][available], xy[available, 0], settings["inclination"]
        )
        absolute = np.abs(conditional[available])
        conditional_stats = dict(p99=float(np.quantile(absolute, 0.99)), max=float(absolute.max()))
        conditional_stats["passed"] = all(
            conditional_stats[k] <= v for k, v in settings["conditional"].items()
        )
        reference = references[str(int(distance))]
        product = reconstruct_rings(
            s,
            distance,
            beta,
            radii,
            reference["b"],
            settings["neighbours"],
            settings["stencil_distance"],
            settings["stencil_radius_span"],
            settings["max_offset"],
        )
        np.savez_compressed(
            destination / f"{name}_analysis.npz",
            **product,
            conditional_residual=conditional,
            hit_mask=hits,
            available_mask=available,
        )
        products[name] = product
        diagnostics = transfer.diagnostics()
        clean = all(
            diagnostics[key] == 0
            for key in [
                "unfinished",
                "invalid_radius",
                "invalid_g",
                "invalid_radiance",
                "missing_g_at_hit",
            ]
        )
        metrics[name] = dict(
            matched=compare(product, reference, settings["matched"]),
            conditional=conditional_stats,
            diagnostics=diagnostics,
            clean=clean,
        )
        write_json(destination / "metrics_partial.json", metrics)
        print(name, metrics[name], flush=True)
    pairs = [
        ("R4000_n3", "R4000_n5"),
        ("R4000_n5", "R4000_n7"),
        ("R4000_n7", "R4000_jitter7"),
        ("R4000_n7", "R4000_tight7"),
    ]
    refinement = {
        a + "_to_" + b: compare(products[a], products[b], settings["refinement"]) for a, b in pairs
    }
    a, b = transfers["R4000_n7"].samples, transfers["R4000_tight7"].samples
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
    finite = {}
    for label in ["1000", "2000", "4000"]:
        finite[label] = [
            {
                key: residual_metrics(references[label][key][j] - references["infinity"][key][j])
                for key in ["b", "z"]
            }
            for j in range(len(radii))
        ]
    distant_pass = all(
        finite["4000"][j][key]["max"] <= settings["distant"][key + "_max"]
        and finite["4000"][j][key]["max"]
        < finite["2000"][j][key]["max"]
        < finite["1000"][j][key]["max"]
        for j in range(len(radii))
        for key in ["b", "z"]
    )
    protected_ok = all(
        sha256(ROOT / name) == digest for name, digest in protocol["protected"].items()
    )
    summary = dict(
        metrics=metrics,
        refinement=refinement,
        tolerance=tolerance,
        reference_checks=reference_checks,
        finite_distance=finite,
        distant_approximation_pass=bool(distant_pass),
        matched_pass=all(metrics[n]["matched"]["passed"] for n in ["R4000_n7", "R4000_jitter7"]),
        refinement_pass=all(row["passed"] for row in refinement.values()),
        conditional_pass=all(row["conditional"]["passed"] for row in metrics.values()),
        clean=all(row["clean"] for row in metrics.values()),
        historical_artifacts_unchanged=protected_ok,
        render_count=len(renders),
        render_seconds=sum(row["seconds"] for row in renders),
        driver_seconds=time.perf_counter() - start,
        artifact_bytes=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
    )
    summary["reproduced"] = (
        all(summary[k] for k in ["matched_pass", "refinement_pass", "conditional_pass", "clean"])
        and tolerance["passed"]
    )
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
