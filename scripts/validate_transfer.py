import argparse
import json
import shutil
import subprocess
import time
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tomli_w

from kerr_sbi.pfm import read_pfm
from kerr_sbi.transfer import BUILD_FIELDS, GeometryCache, Transfer, geometry_key, sha256
from kerr_sbi.visibility import FourierImage, direct_visibility, load_eht_csv, visibility_distance

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def scene(
    spin: float,
    inclination: float,
    radius: float,
    q: float,
    n: int,
    nmax: int,
    jitter: bool,
    width: int,
    height: int,
    tol: float,
) -> dict:
    angle = np.deg2rad(inclination)
    return {
        "metric": {"kind": "kerr", "mass": 1.0, "spin": spin},
        "camera": {
            "position": [-85 * np.sin(angle), 0.0, 85 * np.cos(angle)],
            "fov_deg": 22.0,
            "width": width,
            "height": height,
            "supersample": n,
            "supersample_max": nmax,
            "jitter": jitter,
        },
        "disk": {
            "model": "stylized",
            "r_in": radius,
            "r_out": 15.0,
            "emissivity_index": q,
            "g_power": 3.0,
        },
        "sky": {"uniform": [0.0, 0.0, 0.0]},
        "integrator": {"tol": tol, "max_steps": 20000},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/transfer_pilot.toml")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("run must be a single directory name")
    settings = tomllib.loads(args.settings.read_text())["experiment"]
    if (
        6 * (2 + len(settings["emissivity_variants"])) + 2 + 2 * (len(settings["sampling"]) + 1)
        > settings["render_budget"]
    ):
        raise ValueError("planned render count exceeds budget")
    destination = ROOT / "runs" / args.run
    destination.mkdir(exist_ok=False)
    identity = json.loads(args.identity.read_text())
    binary = Path(identity["binary_path"])
    cache = GeometryCache(destination / "geometry", binary, identity)
    csv = ROOT / "runs/eht_visibility_20261003/observations.csv"
    prior = ROOT / "runs/eht_emission_20261003/summary.json"
    selected = json.loads(prior.read_text())
    radius, inclination, q, scale, angle, dx, dy = selected["parameters"]
    sources = {
        "truth": dict(
            spin=0.98,
            inclination=18.0,
            radius=8.0,
            q=2.0,
            scale=1.0,
            angle=0.0,
            offset=[0.0, 0.0],
            flux=0.6,
        ),
        "candidate": dict(
            spin=0.0,
            inclination=inclination,
            radius=radius,
            q=q,
            scale=scale,
            angle=angle,
            offset=[dx, dy],
            flux=selected["flux_jy"],
        ),
    }
    protected = [ROOT / ".tools/nullgeo/bin/nullgeo"]
    for directory in ROOT.glob("runs/eht_*_20261003"):
        protected.extend(p for p in directory.rglob("*") if p.is_file())
    preserved = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    snapshots = [
        Path(__file__),
        args.settings,
        ROOT / "src/kerr_sbi/transfer.py",
        ROOT / "src/kerr_sbi/visibility.py",
        ROOT / "src/kerr_sbi/pfm.py",
        ROOT / "uv.lock",
        args.identity,
    ]
    for path in snapshots:
        shutil.copyfile(path, destination / path.name)
    shutil.copyfile(csv, destination / "observations.csv")
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "settings": settings,
        "sources": sources,
        "consumer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "consumer_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "binary": identity,
        "csv_sha256": sha256(csv),
        "selected_source_sha256": sha256(prior),
        "source_hashes": {str(p): sha256(p) for p in snapshots},
        "preserved_artifacts": preserved,
        "acceptance": [
            "Stored-radiance PFM reconstruction exact; computed emission within "
            "8*(Nmax+2)*eps32*peak; report actual errors.",
            "All six layouts must pass; adaptive cases must include base and replacement pixels.",
            "No invalid emission or unfinished scientific rays; preserve failures in schema tests.",
            "Require each source sampling comparison and tolerance comparison D < 0.25.",
            "Require pixel FFT versus pixel direct D < 0.05; algebra error < 2e-13 Jy.",
            "Direct-subray versus uniform-pixel differences are representation differences, "
            "not a convergence estimate and have no pass/fail gate.",
            "Both sampling steps and tighter tolerance must pass for selected-case suitability; "
            "passing is not a continuum proof or population validation.",
        ],
        "conventions": "East=screen right, north=screen up; positive rotation east toward north; "
        "negative Fourier phase for positive sky position; assigned horizontal "
        "125 uas at finite camera radius 85M, flat screen weights, no subray sinc. "
        "D^2=sum(abs(delta V)^2/Isigma^2), independent noise per complex component.",
        "restrictions": "Fixed disk edges in each cache, ISCO clamp retained. No nuisance fitting, "
        "training, reserved test split, plunging emission or physical calibration.",
    }
    write_json(destination / "protocol.json", protocol)
    start = time.perf_counter()
    records = []
    results = {"reconstruction": {}, "failures": {}, "pilot": {}, "comparisons": {}}
    predictions = {}
    data = load_eht_csv(csv, settings["minimum_baseline_lambda"])

    def guard() -> None:
        if len(records) >= settings["render_budget"]:
            raise RuntimeError("render budget exhausted")
        if time.perf_counter() - start >= settings["max_seconds"]:
            raise RuntimeError("wall time budget exhausted")
        if (
            sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
            > settings["storage_budget_bytes"]
        ):
            raise RuntimeError("storage budget exhausted")

    def record(name: str, seconds: float, command: list[str]) -> None:
        records.append(dict(name=name, seconds=seconds, command=command))
        write_json(destination / "renders.json", records)
        print(
            f"Render {len(records)}/{settings['render_budget']}: {name} ({seconds:.3f}s)",
            flush=True,
        )

    def exported(name: str, config: dict) -> tuple[Transfer, Path]:
        guard()
        transfer = cache.get(config, timeout=settings["render_timeout_seconds"])
        directory = cache.root / geometry_key(config, identity)
        entry = json.loads((directory / "cache.json").read_text())
        record(name, entry["seconds"], entry["command"])
        return transfer, directory / "image.pfm"

    def ordinary(name: str, config: dict) -> Path:
        guard()
        folder = destination / "ordinary"
        folder.mkdir(exist_ok=True)
        path, pfm = folder / f"{name}.toml", folder / f"{name}.pfm"
        config = deepcopy(config)
        config["output"] = [{"path": str(pfm), "format": "pfm"}]
        path.write_text(tomli_w.dumps(config))
        command = [str(binary), "render", str(path)]
        before = time.perf_counter()
        with (folder / f"{name}.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=settings["render_timeout_seconds"],
            )
        record(name, time.perf_counter() - before, command)
        return pfm

    def compare(transfer: Transfer, pfm: Path, q: float, power: float) -> dict:
        rgb = read_pfm(pfm)
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])
        pixels = transfer.pixels(transfer.intensity(q, power))
        error = pixels.astype(float) - rgb[:, :, 0]
        nmax = max(np.bincount(transfer.samples["pixel_index"].astype(np.intp)))
        bound = (
            settings["reconstruction_roundoff_multiplier"]
            * (nmax + 2)
            * np.finfo(np.float32).eps
            * max(float(rgb.max()), 1e-30)
        )
        result = dict(
            max_abs=float(np.max(np.abs(error))),
            relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(rgb[:, :, 0]), 1e-30)),
            tolerance_abs=bound,
        )
        if result["max_abs"] > bound:
            raise ValueError(f"reconstruction failed: {result}")
        return result

    for name, n, nmax, jitter in [
        ("single", 1, 1, False),
        ("single_jitter", 1, 1, True),
        ("grid", 3, 3, False),
        ("jitter", 3, 3, True),
        ("adaptive", 1, 3, False),
        ("adaptive_jitter", 2, 3, True),
    ]:
        config = scene(0.7, 45.0, 8.0, 2.0, n, nmax, jitter, 32, 24, 1e-8)
        transfer, pfm = exported(name, config)
        diagnostics = transfer.diagnostics()
        if nmax > n and len(diagnostics["pixels_by_sample_count"]) != 2:
            raise ValueError("adaptive fixture did not exercise replacement")
        np.testing.assert_array_equal(transfer.pixels(), read_pfm(pfm)[:, :, 0])
        plain = ordinary(f"{name}_original", config)
        if pfm.read_bytes() != plain.read_bytes():
            raise ValueError("export changed ordinary PFM")
        entry = dict(
            diagnostics=diagnostics, original=compare(transfer, pfm, 2.0, 3.0), variants=[]
        )
        for q_value, power in settings["emissivity_variants"]:
            changed = deepcopy(config)
            changed["disk"].update(emissivity_index=q_value, g_power=power)
            variant = ordinary(f"{name}_q{q_value}_g{power}", changed)
            entry["variants"].append(
                dict(q=q_value, g_power=power, **compare(transfer, variant, q_value, power))
            )
            cached = cache.get(changed)
            np.testing.assert_array_equal(cached.samples["radiance"], transfer.samples["radiance"])
        results["reconstruction"][name] = entry
        write_json(destination / "partial.json", results)

    for name, steps, tol, outcome in [("max_steps", 0, 1e-8, 4), ("stalled", 20000, 1e-300, 5)]:
        config = scene(0.7, 45.0, 8.0, 2.0, 1, 1, False, 32, 24, tol)
        config["integrator"]["max_steps"] = steps
        transfer, pfm = exported(name, config)
        if not np.all(transfer.samples["outcome"] == outcome):
            raise ValueError("failure termination flags mismatch")
        np.testing.assert_array_equal(transfer.pixels(), read_pfm(pfm)[:, :, 0])
        try:
            transfer.visibility(data.uv[:1], 125)
        except ValueError as error:
            if "unfinished" not in str(error):
                raise
        else:
            raise ValueError("unfinished scientific visibility was accepted")
        results["failures"][name] = transfer.diagnostics()

    for source_name, source in sources.items():
        modes = [(f"n{n}", n, settings["tolerance"]) for n in settings["sampling"]]
        modes += [("tight", settings["sampling"][-1], settings["tight_tolerance"])]
        for mode, n, tol in modes:
            name = f"{source_name}_{mode}"
            config = scene(
                source["spin"],
                source["inclination"],
                source["radius"],
                source["q"],
                n,
                n,
                False,
                64,
                64,
                tol,
            )
            transfer, pfm = exported(name, config)
            diagnostics = transfer.diagnostics()
            if any(
                diagnostics[key]
                for key in (
                    "unfinished",
                    "invalid_radius",
                    "invalid_g",
                    "invalid_radiance",
                    "missing_g_at_hit",
                )
            ):
                raise ValueError("scientific sample quality gate failed")
            np.testing.assert_array_equal(transfer.pixels(), read_pfm(pfm)[:, :, 0])
            reconstruction = compare(transfer, pfm, source["q"], 3.0)
            kwargs = dict(
                fov_uas=settings["fov_uas"] * source["scale"],
                flux_jy=source["flux"],
                angle_deg=source["angle"],
                offset_uas=tuple(source["offset"]),
                baseline_batch=settings["baseline_batch"],
                sample_batch=settings["sample_batch"],
            )
            before = time.perf_counter()
            vis = transfer.visibility(data.uv, **kwargs)
            seconds = time.perf_counter() - before
            check = transfer.visibility(
                np.vstack(([0.0, 0.0], data.uv[:8], -data.uv[:8])), **kwargs
            )
            zero_error = float(abs(check[0] - source["flux"]))
            conjugacy_error = float(np.max(np.abs(check[1:9] - check[9:].conj())))
            if max(zero_error, conjugacy_error) >= settings["max_algebra_error_jy"]:
                raise ValueError("visibility algebra gate failed")
            pixel = kwargs["fov_uas"] / 64
            image = read_pfm(pfm)[:, :, 0]
            uniform = source["flux"] * direct_visibility(
                image, data.uv, pixel, source["angle"], tuple(source["offset"])
            )
            fft = source["flux"] * FourierImage(image).evaluate(
                data.uv, pixel, source["angle"], tuple(source["offset"])
            )
            entry = dict(
                diagnostics=diagnostics,
                reconstruction=reconstruction,
                visibility_seconds=seconds,
                zero_error_jy=zero_error,
                conjugacy_error_jy=conjugacy_error,
                representation_D=visibility_distance(vis, uniform, data.sigma),
                fft_D=visibility_distance(fft, uniform, data.sigma),
            )
            results["pilot"][name] = entry
            predictions[name] = vis
            predictions[name + "_pixel"] = uniform
            print(
                f"{name}: subray/pixel D={entry['representation_D']:.6f}, "
                f"visibility {seconds:.2f}s",
                flush=True,
            )
            write_json(destination / "partial.json", results)
            np.savez(destination / "predictions.npz", uv=data.uv, sigma=data.sigma, **predictions)
            write_json(
                destination / "simulation.json",
                {"binary": identity, "build": {k: transfer.metadata[k] for k in BUILD_FIELDS}},
            )

    sampling_errors, tolerance_errors = [], []
    for source_name in sources:
        entry = {}
        for left, right in zip(settings["sampling"][:-1], settings["sampling"][1:], strict=True):
            a, b = f"{source_name}_n{left}", f"{source_name}_n{right}"
            error = visibility_distance(predictions[a], predictions[b], data.sigma)
            sampling_errors.append(error)
            entry[f"{left}_to_{right}"] = dict(
                subray_D=error,
                pixel_D=visibility_distance(
                    predictions[a + "_pixel"], predictions[b + "_pixel"], data.sigma
                ),
            )
        a, b = f"{source_name}_n{settings['sampling'][-1]}", f"{source_name}_tight"
        error = visibility_distance(predictions[a], predictions[b], data.sigma)
        tolerance_errors.append(error)
        entry["tolerance"] = dict(
            subray_D=error,
            pixel_D=visibility_distance(
                predictions[a + "_pixel"], predictions[b + "_pixel"], data.sigma
            ),
        )
        results["comparisons"][source_name] = entry
    results["pair_distances"] = {
        mode: {
            kind: visibility_distance(
                predictions["truth_" + mode + suffix],
                predictions["candidate_" + mode + suffix],
                data.sigma,
            )
            for kind, suffix in [("subray", ""), ("pixel", "_pixel")]
        }
        for mode in [f"n{n}" for n in settings["sampling"]] + ["tight"]
    }
    results["sampling_passed"] = max(sampling_errors) < settings["max_sampling_noise_norm"]
    results["tolerance_passed"] = max(tolerance_errors) < settings["max_tolerance_noise_norm"]
    results["fft_passed"] = (
        max(e["fft_D"] for e in results["pilot"].values()) < settings["max_fft_noise_norm"]
    )
    results["selected_case_suitable"] = all(
        results[k] for k in ("sampling_passed", "tolerance_passed", "fft_passed")
    )
    results["render_count"] = len(records)
    results["render_seconds"] = sum(record["seconds"] for record in records)
    results["wall_seconds"] = time.perf_counter() - start
    results["measurements"] = len(data.sigma)
    results["geometry_bytes"] = sum(p.stat().st_size for p in cache.root.rglob("*") if p.is_file())
    results["historical_artifacts_unchanged"] = all(
        sha256(ROOT / name) == digest for name, digest in preserved.items()
    )
    if not results["historical_artifacts_unchanged"]:
        raise ValueError("historical artifact changed")
    write_json(destination / "summary.json", results)
    print(
        json.dumps(
            {k: v for k, v in results.items() if k not in ("reconstruction", "pilot")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
