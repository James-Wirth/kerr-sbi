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
from scipy.ndimage import binary_dilation
from validate_transfer import scene, write_json

from kerr_sbi.pfm import read_pfm
from kerr_sbi.transfer import GeometryCache, Transfer, geometry_key, sha256
from kerr_sbi.transfer_fourier import TransferFourier
from kerr_sbi.visibility import FourierImage, direct_visibility, load_eht_csv, visibility_distance

ROOT = Path(__file__).resolve().parents[1]


def region_masks(transfers: list[Transfer], dilation: int, span: float) -> dict:
    shape = (transfers[0].metadata["height"], transfers[0].metadata["width"])
    boundary, branch, emitting = [np.zeros(np.prod(shape), dtype=bool) for _ in range(3)]
    for transfer in transfers:
        s = transfer.samples
        ids = s["pixel_index"].astype(np.intp)
        count = np.bincount(ids)
        hits = np.bincount(ids, weights=s["has_intersection"])
        boundary |= (hits > 0) & (hits < count)
        emitting |= hits > 0
        minimum = np.full(len(count), np.inf)
        maximum = np.full(len(count), -np.inf)
        valid = s["has_intersection"].astype(bool)
        np.minimum.at(minimum, ids[valid], s["radius"][valid])
        np.maximum.at(maximum, ids[valid], s["radius"][valid])
        branch |= maximum - minimum > span
    boundary = binary_dilation(boundary.reshape(shape), iterations=dilation).ravel()
    branch = binary_dilation(branch.reshape(shape), iterations=dilation).ravel() & ~boundary
    return dict(
        boundary=boundary,
        branch=branch,
        smooth=emitting & ~boundary & ~branch,
        dark=~emitting & ~boundary & ~branch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/transfer_refinement.toml")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("run must be a single new directory name")
    start = time.perf_counter()
    destination = ROOT / "runs" / args.run
    destination.mkdir(exist_ok=False)
    old = ROOT / "runs/transfer_pilot_20261003"
    prior = json.loads((old / "protocol.json").read_text())
    settings = tomllib.loads(args.settings.read_text())["experiment"]
    if (
        settings["sampling"] != [25, 35, 49]
        or settings["pattern_sampling"] != 49
        or settings["tolerance"] != 1e-9
        or settings["render_budget"] != 6
        or settings["max_seconds"] > 2400
        or settings["storage_budget_bytes"] > 5900000000
    ):
        raise ValueError("settings differ from this bounded study design")

    def deadline(signum: int, frame: object) -> None:
        raise TimeoutError("bounded study computation deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, settings["max_seconds"])
    identity, sources = prior["binary"], prior["sources"]
    if (
        identity["binary_sha256"]
        != "722f2a73b83fd05931a73a86b007476c92d257496b7a1db5e133361cb0aa786b"
    ):
        raise ValueError("unexpected renderer")
    csv = old / "observations.csv"
    if sha256(csv) != prior["csv_sha256"]:
        raise ValueError("CSV identity mismatch")
    data = load_eht_csv(csv, prior["settings"]["minimum_baseline_lambda"])
    saved = np.load(old / "predictions.npz")
    np.testing.assert_array_equal(data.uv, saved["uv"])
    np.testing.assert_array_equal(data.sigma, saved["sigma"])
    old_cache = GeometryCache(old / "geometry", Path(identity["binary_path"]), identity)
    cache = GeometryCache(destination / "geometry", Path(identity["binary_path"]), identity)
    protected = {}
    for folder in [old, *ROOT.glob("runs/eht_*_20261003"), ROOT / ".tools"]:
        for path in folder.rglob("*"):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in prior["preserved_artifacts"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"historical identity mismatch: {name}")
    snapshots = [
        Path(__file__),
        args.settings,
        ROOT / "scripts/validate_transfer.py",
        ROOT / "src/kerr_sbi/transfer_fourier.py",
        ROOT / "src/kerr_sbi/transfer.py",
        ROOT / "src/kerr_sbi/visibility.py",
        ROOT / "uv.lock",
    ]
    for path in snapshots:
        shutil.copyfile(path, destination / path.name)
    protocol = dict(
        created_utc=datetime.now(UTC).isoformat(),
        settings=settings,
        sources=sources,
        binary=identity,
        prior_protocol_sha256=sha256(old / "protocol.json"),
        prior_predictions_sha256=sha256(old / "predictions.npz"),
        csv_sha256=sha256(csv),
        consumer_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in snapshots},
        preserved_artifacts=protected,
        plan="For each source: reuse verified 15,25,tight25; new regular35,regular49,jitter49. "
        "No contingent extra renders. Two successive 25->35->49 refinements plus "
        "regular49 versus deterministic jitter49 must EACH satisfy D<0.25. "
        "Jitter repeats the same base-2/base-3 radical-inverse pattern in every pixel; "
        "this is a distinct pattern check, not independent random replicates.",
        acceleration="Degree-12 total-degree Taylor expansion of subpixel Fourier phase; "
        "bound per baseline = total flux * [pi*(abs(fx)+abs(fy))]^13/13!. "
        "This bounds truncation only, not floating-point error. Require bound "
        "noise norm and measured direct-comparison norm each <1e-7. "
        "Compare both sources' old 15,25,tight25 against saved ALL-baseline "
        "direct sums; recompute direct on 32 evenly spaced baselines per export, "
        "and full 6568-baseline direct sums for both new jitter49 exports.",
        diagnosis="Fixed pixel regions from union of old15/25 partial-hit pixels, dilated "
        "by one pixel; branch proxy = hit radius span >2M, dilated one pixel, "
        "excluding boundary region. Remaining emitting pixels are smooth-region "
        "proxy; others dark. No region is removed from scientific prediction. "
        "Decompose comparisons in common right-hand flux normalization and report "
        "separate global normalization term, with additive closure. Same-node "
        "25/tight25 checks report mask changes and radius/g sensitivity. "
        "Boundary attribution is a hypothesis; regions are proxies, not image orders.",
        acceptance="Every individual-source refinement and pattern D<0.25; existing same-node "
        "25/tight25 sensitivity D<0.25; full-baseline uniform-pixel FFT/direct D<0.05; "
        "zero flux/conjugacy error <2e-13 Jy; exact stored-radiance reconstruction; "
        "no invalid/missing emission or unfinished rays. No threshold relaxation.",
        conventions=prior["conventions"],
        restrictions=prior["restrictions"],
        budget="Six new sequential renders, 2400s driver wall ceiling (300s reserved within "
        "the authorized 2700s computation budget), 5.9GB driver storage ceiling "
        "within authorized 6GB. Stop unresolved if any budget prevents completion.",
    )
    write_json(destination / "protocol.json", protocol)
    print("Protocol frozen before new renders", flush=True)
    records, predictions, results, models = [], {}, {}, {}
    diagnostics = {}

    def guard(reserve_seconds: float = 0, reserve_bytes: int = 0) -> None:
        if time.perf_counter() - start + reserve_seconds >= settings["max_seconds"]:
            raise RuntimeError("computation budget exhausted")
        size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        if size + reserve_bytes > settings["storage_budget_bytes"]:
            raise RuntimeError("artifact budget exhausted")

    def config(source: dict, n: int, jitter: bool = False, tol: float = 1e-9) -> dict:
        return scene(
            source["spin"],
            source["inclination"],
            source["radius"],
            source["q"],
            n,
            n,
            jitter,
            64,
            64,
            tol,
        )

    def existing(source: dict, n: int, tol: float = 1e-9) -> Transfer:
        requested = config(source, n, tol=tol)
        if not (old_cache.root / geometry_key(requested, identity) / "cache.json").is_file():
            raise ValueError("required reuse missing; retrace forbidden")
        return old_cache.get(requested)

    def evaluate(name: str, source: dict, transfer: Transfer, folder: Path, full: bool) -> None:
        guard()
        before = time.perf_counter()
        quality = transfer.diagnostics()
        if any(
            quality[k]
            for k in (
                "unfinished",
                "invalid_radius",
                "invalid_g",
                "invalid_radiance",
                "missing_g_at_hit",
            )
        ):
            raise ValueError("scientific sample quality gate failed")
        image = read_pfm(folder / "image.pfm")[:, :, 0]
        np.testing.assert_array_equal(transfer.pixels(), image)
        emitted_error = float(
            np.max(abs(transfer.pixels(transfer.intensity(source["q"], 3)) - image))
        )
        allowance = (
            8
            * (max(map(int, quality["pixels_by_sample_count"])) + 2)
            * np.finfo(np.float32).eps
            * image.max()
        )
        if emitted_error > allowance:
            raise ValueError("emission reconstruction gate failed")
        kwargs = dict(
            fov_uas=125 * source["scale"],
            flux_jy=source["flux"],
            angle_deg=source["angle"],
            offset_uas=tuple(source["offset"]),
        )
        model = TransferFourier(transfer, source["q"], 3, order=settings["moment_order"])
        constructed = time.perf_counter()
        vis, bound = model.evaluate(data.uv, **kwargs)
        computed = time.perf_counter()
        bounds_D = float(np.linalg.norm(bound / data.sigma))
        if bounds_D >= settings["max_acceleration_noise_norm"]:
            raise ValueError("acceleration bound gate failed")
        indices = np.unique(np.r_[np.linspace(0, len(data.uv) - 1, 32, dtype=int)])
        direct = transfer.visibility(data.uv if full else data.uv[indices], **kwargs)
        measured = visibility_distance(
            vis if full else vis[indices], direct, data.sigma if full else data.sigma[indices]
        )
        if measured >= settings["max_acceleration_noise_norm"]:
            raise ValueError("acceleration comparison gate failed")
        saved_D = visibility_distance(vis, saved[name], data.sigma) if name in saved else None
        if saved_D is not None and saved_D >= settings["max_acceleration_noise_norm"]:
            raise ValueError("saved direct comparison gate failed")
        checks, _ = model.evaluate(
            np.vstack(([0, 0], data.uv[indices], -data.uv[indices])), **kwargs
        )
        zero = float(abs(checks[0] - source["flux"]))
        symmetry = float(
            np.max(abs(checks[1 : 1 + len(indices)] - checks[1 + len(indices) :].conj()))
        )
        if max(zero, symmetry) >= settings["max_algebra_error_jy"]:
            raise ValueError("algebra gate failed")
        pixel = kwargs["fov_uas"] / 64
        uniform = source["flux"] * direct_visibility(
            image, data.uv, pixel, source["angle"], tuple(source["offset"])
        )
        fft = source["flux"] * FourierImage(image).evaluate(
            data.uv, pixel, source["angle"], tuple(source["offset"])
        )
        fft_D = visibility_distance(uniform, fft, data.sigma)
        if fft_D >= settings["max_fft_noise_norm"]:
            raise ValueError("pixel Fourier gate failed")
        results[name] = dict(
            quality=quality,
            reconstruction_max_abs=emitted_error,
            moment_seconds=constructed - before,
            evaluation_seconds=computed - constructed,
            total_seconds=time.perf_counter() - before,
            truncation_bound_D=bounds_D,
            measured_direct_D=measured,
            direct_baselines=len(data.uv) if full else len(indices),
            saved_direct_D=saved_D,
            zero_error_jy=zero,
            conjugacy_error_jy=symmetry,
            fft_D=fft_D,
            representation_D=visibility_distance(vis, uniform, data.sigma),
            total_screen_mass=model.total_mass,
        )
        predictions[name], models[name] = vis, model
        np.savez(
            destination / f"{name}_moments.npz",
            moments=model.moments,
            centers=model.centers,
            total_mass=model.total_mass,
            order=model.order,
        )
        np.savez(destination / "predictions.npz", uv=data.uv, sigma=data.sigma, **predictions)
        write_json(destination / "partial.json", results)
        print(
            f"{name}: evaluation {computed - constructed:.2f}s, direct D={measured:.3g}", flush=True
        )
        guard()

    for source_name, source in sources.items():
        old_transfers = [existing(source, n) for n in (15, 25)]
        masks = region_masks(
            old_transfers, settings["boundary_dilation_pixels"], settings["branch_radius_span"]
        )
        np.savez(destination / f"{source_name}_regions.npz", **masks)
        for mode, transfer in zip(("n15", "n25"), old_transfers, strict=True):
            folder = old_cache.root / geometry_key(transfer.scene, identity)
            evaluate(f"{source_name}_{mode}", source, transfer, folder, False)
        tight = existing(source, 25, tol=1e-11)
        evaluate(
            f"{source_name}_tight",
            source,
            tight,
            old_cache.root / geometry_key(tight.scene, identity),
            False,
        )
        regular = old_transfers[-1].samples
        for field in ("screen_u", "screen_v", "weight", "pixel_index", "sample_index"):
            np.testing.assert_array_equal(regular[field], tight.samples[field])
        same = (regular["has_intersection"] == 1) & (tight.samples["has_intersection"] == 1)
        diagnostics[source_name] = dict(
            region_pixels={k: int(v.sum()) for k, v in masks.items()},
            tolerance_mask_changes=int(
                np.sum(regular["has_intersection"] != tight.samples["has_intersection"])
            ),
            tolerance_outcome_changes=int(np.sum(regular["outcome"] != tight.samples["outcome"])),
            tolerance_radius_max=float(
                np.max(abs(regular["radius"][same] - tight.samples["radius"][same]))
            ),
            tolerance_g_max=float(np.max(abs(regular["g"][same] - tight.samples["g"][same]))),
        )
        del old_transfers, tight, regular
        for mode, n, jitter in [("n35", 35, False), ("n49", 49, False), ("jitter49", 49, True)]:
            guard(settings["render_timeout_seconds"], 4096 * n * n * 103 + 10000000)
            if len(records) >= settings["render_budget"]:
                raise RuntimeError("render budget exhausted")
            requested = config(source, n, jitter)
            before = time.perf_counter()
            records.append(dict(name=f"{source_name}_{mode}", status="started"))
            write_json(destination / "renders.json", records)
            transfer = cache.get(
                requested,
                timeout=min(
                    settings["render_timeout_seconds"],
                    settings["max_seconds"] - (time.perf_counter() - start),
                ),
            )
            folder = cache.root / geometry_key(requested, identity)
            records[-1].update(
                status="completed",
                seconds=time.perf_counter() - before,
                command=json.loads((folder / "cache.json").read_text())["command"],
            )
            write_json(destination / "renders.json", records)
            print(
                f"Render {len(records)}/6: {source_name}_{mode}, {records[-1]['seconds']:.1f}s",
                flush=True,
            )
            evaluate(f"{source_name}_{mode}", source, transfer, folder, jitter)
            del transfer
        comparisons = {}
        kwargs = dict(
            fov_uas=125 * source["scale"],
            flux_jy=source["flux"],
            angle_deg=source["angle"],
            offset_uas=tuple(source["offset"]),
        )
        for left, right in [
            ("n15", "n25"),
            ("n25", "n35"),
            ("n35", "n49"),
            ("n49", "jitter49"),
            ("n25", "tight"),
        ]:
            a, b = models[f"{source_name}_{left}"], models[f"{source_name}_{right}"]
            va, vb = predictions[f"{source_name}_{left}"], predictions[f"{source_name}_{right}"]
            common = dict(kwargs, flux_jy=source["flux"] * a.total_mass / b.total_mass)
            ca, _ = a.evaluate(data.uv, **common)
            pieces = dict(normalization=va - ca)
            for region, mask in masks.items():
                x, _ = a.evaluate(data.uv, pixel_mask=mask, **common)
                y, _ = b.evaluate(data.uv, pixel_mask=mask, **kwargs)
                pieces[region] = x - y
            closure = visibility_distance(sum(pieces.values()), va - vb, data.sigma)
            comparisons[f"{left}_to_{right}"] = dict(
                D=visibility_distance(va, vb, data.sigma),
                components_D={k: float(np.linalg.norm(v / data.sigma)) for k, v in pieces.items()},
                closure_D=closure,
                relative_mass_change=a.total_mass / b.total_mass - 1,
                component_inner_products={
                    k: float(np.vdot(v / data.sigma, (va - vb) / data.sigma).real)
                    for k, v in pieces.items()
                },
            )
            np.savez(destination / f"{source_name}_{left}_to_{right}_components.npz", **pieces)
        diagnostics[source_name]["comparisons"] = comparisons
        write_json(destination / "diagnostics.json", diagnostics)
        print(json.dumps({source_name: comparisons}, indent=2), flush=True)
    gates = [
        diagnostics[s]["comparisons"][mode]["D"]
        for s in sources
        for mode in ("n25_to_n35", "n35_to_n49", "n49_to_jitter49", "n25_to_tight")
    ]
    summary = dict(
        selected_case_suitable=all(d < settings["max_sampling_noise_norm"] for d in gates),
        comparisons=diagnostics,
        exports=results,
        render_count=len(records),
        render_seconds=sum(r["seconds"] for r in records),
        measurements=len(data.uv),
        pair_distances={
            m: visibility_distance(
                predictions[f"truth_{m}"], predictions[f"candidate_{m}"], data.sigma
            )
            for m in ("n15", "n25", "n35", "n49", "jitter49", "tight")
        },
        historical_artifacts_unchanged=all(sha256(ROOT / p) == h for p, h in protected.items()),
        wall_seconds=time.perf_counter() - start,
        artifact_bytes=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
    )
    guard()
    if not summary["historical_artifacts_unchanged"]:
        raise ValueError("historical artifact changed")
    write_json(destination / "summary.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("exports", "comparisons")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
