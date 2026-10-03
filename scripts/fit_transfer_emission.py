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
from scipy.optimize import least_squares
from validate_transfer import scene, write_json

from kerr_sbi.pfm import read_pfm
from kerr_sbi.radial_emission import RadialFourier, radial_intensity
from kerr_sbi.transfer import GeometryCache, geometry_key, require, sha256
from kerr_sbi.transfer_fourier import TransferFourier
from kerr_sbi.visibility import load_eht_csv, visibility_distance

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("n25", "n35", "n49", "jitter49", "tight")
COMPARISONS = (("n25", "n35"), ("n35", "n49"), ("n49", "jitter49"), ("n25", "tight"))


def nuisance(source: dict) -> list[float]:
    return [source["scale"], source["angle"], *source["offset"], source["flux"]]


def kwargs(parameters: list[float]) -> dict:
    return dict(
        fov_uas=125 * parameters[0],
        angle_deg=parameters[1],
        offset_uas=tuple(parameters[2:4]),
        flux_jy=parameters[4],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transfer_emission.toml")
    args = parser.parse_args()
    require(Path(args.run).name == args.run, "a fresh single directory name is required")
    started = time.perf_counter()
    destination = ROOT / "runs" / args.run
    destination.mkdir(exist_ok=False)
    cfg = tomllib.loads(args.config.read_text())["experiment"]
    require(cfg["new_render_budget"] == 0, "this design is cache-only")
    require(cfg["max_seconds"] <= 2400, "computation ceiling exceeded")
    require(cfg["storage_budget_bytes"] <= 5900000000, "storage ceiling exceeded")

    def deadline(signum: int, frame: object) -> None:
        raise TimeoutError("predeclared computation ceiling reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, cfg["max_seconds"])
    pilot = ROOT / "runs/transfer_pilot_20261003"
    refinement = ROOT / "runs/transfer_refinement_20261003"
    prior = json.loads((refinement / "protocol.json").read_text())
    identity_path = ROOT / ".tools/nullgeo-transfer-c2a46a6-20261003/identity.json"
    identity = json.loads(identity_path.read_text())
    require(identity == prior["binary"], "renderer identity differs from refinement")
    require(sha256(Path(identity["binary_path"])) == identity["binary_sha256"], "binary mismatch")
    sources = prior["sources"]
    csv_path = pilot / "observations.csv"
    require(sha256(csv_path) == prior["csv_sha256"], "measurement identity mismatch")
    data = load_eht_csv(csv_path, 1e8)
    saved = np.load(refinement / "predictions.npz")
    np.testing.assert_array_equal(data.uv, saved["uv"])
    np.testing.assert_array_equal(data.sigma, saved["sigma"])
    require(len(data.uv) == 6568, "incorrect baseline selection")
    protected = dict(prior["preserved_artifacts"])
    for folder in (refinement,):
        for path in folder.rglob("*"):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in protected.items():
        require(sha256(ROOT / name) == digest, f"changed historical artifact: {name}")
    snapshot_paths = [
        Path(__file__),
        args.config,
        ROOT / "scripts/validate_transfer.py",
        ROOT / "src/kerr_sbi/transfer.py",
        ROOT / "src/kerr_sbi/transfer_fourier.py",
        ROOT / "src/kerr_sbi/radial_emission.py",
        ROOT / "tests/test_radial_emission.py",
        ROOT / "src/kerr_sbi/visibility.py",
        ROOT / "uv.lock",
        identity_path,
    ]
    for path in snapshot_paths:
        shutil.copyfile(path, destination / path.name)
    protocol = dict(
        created_utc=datetime.now(UTC).isoformat(),
        settings=cfg,
        sources=sources,
        binary=identity,
        consumer_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in snapshot_paths},
        prior_protocol_sha256=sha256(refinement / "protocol.json"),
        csv_sha256=sha256(csv_path),
        protected_files=protected,
        geometry="Two existing traced geometries only. No new geometry or render. All retained "
        "25,35,49,jitter49,tight25 caches are hash-verified; absent cache is an error.",
        emission="I=g^3*(r/r_in)^(-q)*sum(c_k B_k^3(t)), t=log(r/r_in)/log(r_out/r_in). "
        "q fixed at the existing source value; c0=1, c1..3 in [0.25,4]. All ones exactly "
        "nest the existing power law. Positive cubic modifier, no additional knots or edges, "
        "no clipping or taper. |d modifier/dt|<=11.25 and |d2 modifier/dt2|<=45 follow "
        "from the coefficient box. This does not optimize a free power-law exponent.",
        fitting="First fit observing nuisances only at regular49. Then fit radial coefficients "
        "with neutral nuisances, with the first fit's nuisances frozen, and jointly. Three "
        "starts per fit, as configured; joint starts pair nuisance/radial starts by index. "
        "Use scipy least_squares, bounded TRF, two-point numerical Jacobian, max_nfev=55, "
        "ftol=xtol=gtol=1e-8; select smallest D across starts including start points. "
        "Global fit-phase deadline 1200 seconds from driver start; retain best evaluated "
        "parameters and label unfinished optimizations. Freeze all four fits before checks. "
        "No refitting at other samplings, no gains, no noise draws, no observed-value fitting.",
        comparisons="2x2 neutral/nuisance-fitted and power-law/cubic, plus cubic with first "
        "fit nuisance held fixed. Neutral = scale1,angle0,offset0,flux0.6. Legacy nuisance "
        "prediction separately recorded. Nuisance ranges match prior local emission study.",
        acceptance="D^2=sum(abs(delta V)^2/Isigma^2) over ALL 6568 baselines. Operational "
        "near-degeneracy D<=1 at every regular25,35,49 and jitter49 setting and every source "
        "must pass D<0.25 for 25->35,35->49,49->jitter49 and reused 25->tight25. "
        "Numerical pass also requires sample quality, Fourier and algebra gates. Select "
        "lowest regular49 D among checked fits; if none pass, label all unresolved.",
        fourier="Order12 reusable linear basis moments, no spatial interpolation. Check "
        "32 evenly indexed measured baselines plus all forecast nodes by direct sample "
        "quadrature for the selected pair at regular49; full 6568-baseline direct quadrature "
        "for the selected candidate at regular49. Other fitted cases/settings get 16 "
        "evenly indexed measured direct checks. Norm of truncation bound and direct error "
        "each <1e-7, zero flux/conjugacy errors <2e-13 Jy. Failures remain unresolved.",
        forecast="Cartesian 1 Glambda grid inside radius12 Glambda, one half-plane including "
        "origin (conjugates plotted only, not double counted). Sigma=0.005 Jy per real/imag "
        "component, illustrative ideal calibration. Re-evaluate all per-source sampling "
        "checks on this complete forecast set, norm<0.25 and each point abs(error)/sigma<0.25. "
        "Also report per-point mask where all source checks pass; no claims outside checked "
        "nodes or about continuous interpolation. All forecast Fourier bounds checked. "
        "If whole-domain gate fails show unresolved nodes explicitly, do not relax gate.",
        conventions=prior["conventions"],
        restrictions="ISCO clamp, opaque first hit, original fixed edges/masks/weights retained. "
        "Finite 85M camera,22deg camera FOV; assigned125uas scale. No inside-ISCO emission, "
        "new training, reserved-test use, actual-source inference or global uniqueness claim.",
        budget="0 planned new renders of authorized12; driver<=2400s leaving300s for tests/QA "
        "within2700s computation ceiling. Artifacts<=5.9GB within6GB. No contingent extensions.",
    )
    write_json(destination / "protocol.json", protocol)
    print("Protocol frozen before optimization; zero new renders", flush=True)
    protocol_hash = sha256(destination / "protocol.json")
    caches = {
        "pilot": GeometryCache(pilot / "geometry", Path(identity["binary_path"]), identity),
        "refinement": GeometryCache(
            refinement / "geometry", Path(identity["binary_path"]), identity
        ),
    }
    extent = cfg["forecast_radius_glambda"]
    grid = np.arange(
        -extent, extent + cfg["forecast_step_glambda"] / 2, cfg["forecast_step_glambda"]
    )
    x, y = np.meshgrid(grid, grid)
    mask = (x * x + y * y <= extent * extent) & ((y > 0) | ((y == 0) & (x >= 0)))
    forecast_uv = np.column_stack((x[mask], y[mask])) * 1e9
    forecast_sigma = np.full(len(forecast_uv), cfg["forecast_sigma_jy"])
    quality, models, transfers = {}, {}, {}

    def load(name: str, setting: str):
        source = sources[name]
        n = 25 if setting == "tight" else int(setting.replace("jitter", "").replace("n", ""))
        request = scene(
            source["spin"],
            source["inclination"],
            source["radius"],
            source["q"],
            n,
            n,
            setting == "jitter49",
            64,
            64,
            1e-11 if setting == "tight" else 1e-9,
        )
        cache = caches["pilot" if setting in ("n25", "tight") else "refinement"]
        folder = cache.root / geometry_key(request, identity)
        require((folder / "cache.json").is_file(), "required cache missing; retrace forbidden")
        transfer = cache.get(request)
        diagnostic = transfer.diagnostics()
        require(
            not any(
                diagnostic[k]
                for k in (
                    "unfinished",
                    "invalid_radius",
                    "invalid_g",
                    "missing_g_at_hit",
                    "invalid_radiance",
                )
            ),
            "sample quality failure",
        )
        np.testing.assert_array_equal(transfer.pixels(), read_pfm(folder / "image.pfm")[:, :, 0])
        key = f"{name}_{setting}"
        quality[key] = dict(
            diagnostic, cache=str(folder.relative_to(ROOT)), reconstruction_max_abs=0.0
        )
        transfers[key] = transfer
        model = (
            TransferFourier(transfer, source["q"], 3, order=cfg["moment_order"])
            if name == "truth"
            else RadialFourier(transfer, source["q"], order=cfg["moment_order"])
        )
        models[key] = model
        np.savez_compressed(destination / f"{key}_moments.npz", moments=model.moments)
        print(f"Verified cache and prepared {key}", flush=True)
        return model

    truth_model = load("truth", "n49")
    radial = load("candidate", "n49")
    target, _ = truth_model.evaluate(data.uv, **kwargs(nuisance(sources["truth"])))
    np.testing.assert_allclose(target, saved["truth_n49"], atol=2e-13, rtol=0)
    neutral = [1.0, 0.0, 0.0, 0.0, 0.6]
    ones = [1.0] * 4
    fits = {}
    lower, upper = cfg["nuisance_lower"], cfg["nuisance_upper"]

    def fit(label: str, vary_nuisance: bool, vary_emission: bool, fixed: list[float]) -> dict:
        starts = []
        for i in range(3):
            starts.append(
                (cfg["nuisance_starts"][i] if vary_nuisance else [])
                + (cfg["radial_starts"][i] if vary_emission else [])
            )
        lo = (lower if vary_nuisance else []) + (
            [cfg["coefficient_lower"]] * 3 if vary_emission else []
        )
        hi = (upper if vary_nuisance else []) + (
            [cfg["coefficient_upper"]] * 3 if vary_emission else []
        )
        scales = ([1, 10, 1, 1, 0.6] if vary_nuisance else []) + ([1] * 3 if vary_emission else [])
        records, best, calls = [], None, 0

        def unpack(p):
            ns = p[:5].tolist() if vary_nuisance else fixed
            cs = [1.0, *p[-3:].tolist()] if vary_emission else ones
            return ns, cs

        def residual(p):
            nonlocal best, calls
            if time.perf_counter() - started >= 1200:
                raise TimeoutError("predeclared fitting phase deadline")
            ns, cs = unpack(p)
            model = radial.combine(cs)
            value, bound = model.evaluate(data.uv, **kwargs(ns))
            require(np.linalg.norm(bound / data.sigma) < cfg["max_fourier_D"], "fit Fourier bound")
            delta = (value - target) / data.sigma
            distance = float(np.linalg.norm(delta))
            if best is None or distance < best[0]:
                best = distance, ns, cs
            calls += 1
            if calls % 50 == 0:
                print(f"{label}: {calls} evaluations; best D={best[0]:.8f}", flush=True)
            return np.r_[delta.real, delta.imag]

        for index, start in enumerate(starts):
            try:
                result = least_squares(
                    residual,
                    start,
                    bounds=(lo, hi),
                    x_scale=scales,
                    max_nfev=cfg["max_evaluations"],
                    ftol=cfg["optimizer_tolerance"],
                    xtol=cfg["optimizer_tolerance"],
                    gtol=cfg["optimizer_tolerance"],
                )
                records.append(
                    dict(
                        start=start,
                        distance=float(np.linalg.norm(result.fun)),
                        success=bool(result.success),
                        message=result.message,
                        nfev=result.nfev,
                        optimality=float(result.optimality),
                    )
                )
                print(f"{label} start{index}: D={np.linalg.norm(result.fun):.8f}", flush=True)
            except TimeoutError as error:
                records.append(dict(start=start, success=False, message=str(error)))
                break
        require(best is not None, "no fit evaluated within budget")
        result = dict(
            distance=best[0],
            nuisance=best[1],
            coefficients=best[2],
            starts=records,
            calls=calls,
            all_starts_succeeded=len(records) == 3 and all(r["success"] for r in records),
        )
        fits[label] = result
        write_json(destination / "fits_partial.json", fits)
        return result

    first = fit("nuisance_only", True, False, neutral)
    fit("emission_only", False, True, neutral)
    fit("emission_fixed_nuisance", False, True, first["nuisance"])
    fit("joint", True, True, neutral)
    frozen = dict(
        created_utc=datetime.now(UTC).isoformat(),
        truth=sources["truth"],
        candidate_geometry=sources["candidate"],
        fits=fits,
        protocol_sha256=protocol_hash,
    )
    write_json(destination / "frozen_candidates.json", frozen)
    frozen_hash = sha256(destination / "frozen_candidates.json")
    print("All candidates frozen before sampling checks", flush=True)
    cases = dict(
        neutral=dict(nuisance=neutral, coefficients=ones),
        legacy=dict(nuisance=nuisance(sources["candidate"]), coefficients=ones),
        **fits,
    )
    predictions, forecasts, checks = {}, {}, {}
    direct_indices = np.linspace(0, len(data.uv) - 1, 16, dtype=int)
    for setting in SETTINGS:
        for name in ("truth", "candidate"):
            key = f"{name}_{setting}"
            if key not in models:
                load(name, setting)
        source_cases = dict(
            truth=dict(nuisance=nuisance(sources["truth"]), coefficients=ones), **cases
        )
        for label, case in source_cases.items():
            key = f"{label}_{setting}"
            is_truth = label == "truth"
            transfer = transfers[f"{'truth' if is_truth else 'candidate'}_{setting}"]
            model = (
                models[f"truth_{setting}"]
                if is_truth
                else models[f"candidate_{setting}"].combine(case["coefficients"])
            )
            kw = kwargs(case["nuisance"])
            value, bound = model.evaluate(data.uv, **kw)
            forecast, fb = model.evaluate(forecast_uv, **kw)
            predictions[key], forecasts[key] = value, forecast
            algebra, _ = model.evaluate(
                np.vstack(([0, 0], data.uv[direct_indices], -data.uv[direct_indices])), **kw
            )
            values = (
                transfer.intensity(sources["truth"]["q"], 3)
                if is_truth
                else radial_intensity(transfer, sources["candidate"]["q"], case["coefficients"])
            )
            direct = transfer.visibility(data.uv[direct_indices], values=values, **kw)
            zero = float(abs(algebra[0] - case["nuisance"][4]))
            conjugacy = float(np.max(abs(algebra[1:17] - algebra[17:].conj())))
            checks[key] = dict(
                truncation_D=float(np.linalg.norm(bound / data.sigma)),
                forecast_truncation_D=float(np.linalg.norm(fb / forecast_sigma)),
                direct_D=visibility_distance(
                    value[direct_indices], direct, data.sigma[direct_indices]
                ),
                zero_error_jy=zero,
                conjugacy_error_jy=conjugacy,
            )
            require(
                max(checks[key][k] for k in ("truncation_D", "forecast_truncation_D", "direct_D"))
                < cfg["max_fourier_D"],
                "Fourier gate failed",
            )
            require(max(zero, conjugacy) < cfg["max_algebra_jy"], "algebra gate failed")
            if setting == "n49":
                image = model.moments[:, 0].reshape(64, 64) / model.total_mass * case["nuisance"][4]
                np.save(destination / f"{label}_image.npy", image)
        write_json(destination / "checks_partial.json", checks)
        print(f"Frozen predictions/checks complete: {setting}", flush=True)
    comparisons, forecast_comparisons, pair_distances, accepted = {}, {}, {}, {}
    for label in ("truth", *cases):
        comparisons[label], forecast_comparisons[label] = {}, {}
        for left, right in COMPARISONS:
            comparison = f"{left}_to_{right}"
            comparisons[label][comparison] = visibility_distance(
                predictions[f"{label}_{left}"], predictions[f"{label}_{right}"], data.sigma
            )
            forecast_comparisons[label][comparison] = visibility_distance(
                forecasts[f"{label}_{left}"], forecasts[f"{label}_{right}"], forecast_sigma
            )
        accepted[label] = all(d < cfg["max_sampling_D"] for d in comparisons[label].values())
        if label != "truth":
            pair_distances[label] = {
                setting: visibility_distance(
                    predictions[f"truth_{setting}"], predictions[f"{label}_{setting}"], data.sigma
                )
                for setting in SETTINGS
            }
    eligible = [label for label in fits if accepted[label] and accepted["truth"]]
    selected = min(eligible or list(fits), key=lambda label: pair_distances[label]["n49"])
    print(f"Selected {selected}; measured-domain numerical pass={bool(eligible)}", flush=True)
    selection = dict(
        label=selected,
        numerical_pass=bool(eligible),
        pair_distances=pair_distances,
        comparisons=comparisons,
        forecast_comparisons=forecast_comparisons,
    )
    write_json(destination / "selection.json", selection)
    np.savez_compressed(
        destination / "predictions.npz", uv=data.uv, sigma=data.sigma, **predictions
    )
    np.savez_compressed(
        destination / "forecasts.npz", uv=forecast_uv, sigma=forecast_sigma, **forecasts
    )
    full_checks = {}
    for label in ("truth", selected):
        is_truth = label == "truth"
        source = sources["truth" if is_truth else "candidate"]
        transfer = transfers[f"{'truth' if is_truth else 'candidate'}_n49"]
        case = dict(nuisance=nuisance(source), coefficients=ones) if is_truth else cases[label]
        values = (
            transfer.intensity(source["q"], 3)
            if is_truth
            else radial_intensity(transfer, source["q"], case["coefficients"])
        )
        idx = (
            np.linspace(0, len(data.uv) - 1, 32, dtype=int) if is_truth else np.arange(len(data.uv))
        )
        uv = np.vstack((data.uv[idx], forecast_uv))
        expected = np.r_[predictions[f"{label}_n49"][idx], forecasts[f"{label}_n49"]]
        sigma = np.r_[data.sigma[idx], forecast_sigma]
        print(f"Direct quadrature {label}: {len(uv)} baselines", flush=True)
        direct = transfer.visibility(uv, values=values, **kwargs(case["nuisance"]))
        full_checks[label] = dict(baselines=len(uv), D=visibility_distance(expected, direct, sigma))
        require(full_checks[label]["D"] < cfg["max_fourier_D"], "full direct gate failed")
        np.savez_compressed(destination / f"{label}_direct.npz", uv=uv, sigma=sigma, direct=direct)
    forecast_error = np.zeros(len(forecast_uv))
    for label in ("truth", selected):
        for left, right in COMPARISONS:
            error = (
                abs(forecasts[f"{label}_{left}"] - forecasts[f"{label}_{right}"]) / forecast_sigma
            )
            forecast_error = np.maximum(forecast_error, error)
    forecast_pass = all(
        value < cfg["max_sampling_D"]
        for label in ("truth", selected)
        for value in forecast_comparisons[label].values()
    )
    np.save(destination / "forecast_pointwise_error.npy", forecast_error)
    require(sha256(destination / "protocol.json") == protocol_hash, "protocol mutated")
    require(sha256(destination / "frozen_candidates.json") == frozen_hash, "frozen fits mutated")
    for name, digest in protected.items():
        require(sha256(ROOT / name) == digest, f"historical artifact changed: {name}")
    summary = dict(
        **selection,
        accepted=accepted,
        fits=fits,
        checks=checks,
        quality=quality,
        full_direct_checks=full_checks,
        forecast_domain_pass=forecast_pass,
        forecast_nodes=len(forecast_uv),
        forecast_pointwise_passing=int(np.sum(forecast_error < cfg["max_sampling_D"])),
        near_degenerate=bool(eligible)
        and all(pair_distances[selected][s] <= cfg["near_degeneracy_D"] for s in SETTINGS),
        measurements=len(data.uv),
        render_count=0,
        historical_artifacts_unchanged=True,
        protocol_sha256=protocol_hash,
        frozen_candidates_sha256=frozen_hash,
        wall_seconds=time.perf_counter() - started,
        artifact_bytes=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
    )
    require(summary["artifact_bytes"] < cfg["storage_budget_bytes"], "artifact budget exceeded")
    write_json(destination / "summary.json", summary)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "label",
                    "numerical_pass",
                    "near_degenerate",
                    "forecast_domain_pass",
                    "wall_seconds",
                    "artifact_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
