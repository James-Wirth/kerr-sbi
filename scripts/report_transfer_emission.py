import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.transforms import Affine2D

from kerr_sbi.radial_emission import radial_basis
from kerr_sbi.visibility import direct_visibility, visibility_distance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    folder = ROOT / "runs" / args.run
    summary = json.loads((folder / "summary.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    measured = np.load(folder / "predictions.npz")
    forecast = np.load(folder / "forecasts.npz")
    label = summary["label"]
    fit = summary["fits"][label]
    sources = protocol["sources"]
    uv, sigma = measured["uv"], measured["sigma"]
    length = np.linalg.norm(uv, axis=1) / 1e9
    truth, candidate = measured["truth_n49"], measured[f"{label}_n49"]
    residual = (candidate - truth) / sigma
    fdifference = forecast[f"{label}_n49"] - forecast["truth_n49"]
    score = abs(fdifference) / forecast["sigma"]
    errors = np.load(folder / "forecast_pointwise_error.npy")
    passed = errors < protocol["settings"]["max_sampling_D"]
    order = np.argsort(np.where(passed, score, -1))[::-1]
    best_forecast = [
        dict(
            uv_glambda=(forecast["uv"][i] / 1e9).tolist(),
            difference_mjy=float(abs(fdifference[i]) * 1000),
            single_baseline_D=float(score[i]),
            maximum_sampling_change_in_sigma=float(errors[i]),
        )
        for i in order[:8]
    ]
    groups = []
    for left, right in zip([0, 2, 4, 6, 8], [2, 4, 6, 8, np.inf], strict=True):
        selected = (length >= left) & (length < right)
        groups.append(
            dict(
                min_glambda=left,
                max_glambda=None if np.isinf(right) else right,
                count=int(selected.sum()),
                D2=float(np.sum(abs(residual[selected]) ** 2)),
            )
        )
    largest = np.argsort(abs(residual))[::-1][:8]
    representation = {}
    for name in ("truth", "legacy"):
        source = sources["truth" if name == "truth" else "candidate"]
        image = np.load(folder / f"{name}_image.npy")
        uniform = source["flux"] * direct_visibility(
            image, uv, 125 * source["scale"] / 64, source["angle"], tuple(source["offset"])
        )
        representation[name] = dict(
            subray_to_uniform_D=visibility_distance(measured[f"{name}_n49"], uniform, sigma)
        )
        representation[f"{name}_uniform"] = uniform
    representation["uniform_legacy_pair_D"] = visibility_distance(
        representation.pop("truth_uniform"), representation.pop("legacy_uniform"), sigma
    )
    diagnostic = dict(
        residual_real_D2=float(np.sum(residual.real**2)),
        residual_imag_D2=float(np.sum(residual.imag**2)),
        baseline_groups=groups,
        largest_measured_residuals=[
            dict(
                index=int(i),
                uv_glambda=(uv[i] / 1e9).tolist(),
                sigma_mjy=float(sigma[i] * 1000),
                residual_in_sigma=float(abs(residual[i])),
            )
            for i in largest
        ],
        best_forecast_nodes=best_forecast,
        forecast_pair_D=float(np.linalg.norm(score)),
        representation=representation,
    )
    (folder / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0), layout="constrained")
    images = [np.load(folder / f"{name}_image.npy") for name in ("truth", label)]
    ns = [[1, 0, 0, 0, 0.6], fit["nuisance"]]
    brightness = [
        image * 1000 / (125 * p[0] / 64) ** 2 for image, p in zip(images, ns, strict=True)
    ]
    norm = PowerNorm(0.45, vmin=0, vmax=max(v.max() for v in brightness))
    for ax, values, p, title in zip(
        axes[0, :2],
        brightness,
        ns,
        ["Synthetic truth · spin 0.98", "Checked candidate · spin 0"],
        strict=True,
    ):
        half = 125 * p[0] / 2
        transform = Affine2D().rotate_deg(p[1]).translate(p[2], p[3]) + ax.transData
        im = ax.imshow(
            values,
            origin="upper",
            extent=(-half, half, -half, half),
            transform=transform,
            norm=norm,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set(
            xlim=(-68, 68),
            ylim=(-68, 68),
            aspect="equal",
            xlabel="East (μas)",
            ylabel="North (μas)",
            title=title,
        )
    fig.colorbar(im, ax=axes[0, :2], shrink=0.72, label="mJy / μas² · power stretch")
    ax = axes[0, 2]
    for source, cs, name, color in [
        (sources["truth"], np.ones(4), "Truth", "#2d63ad"),
        (sources["candidate"], fit["coefficients"], "Candidate", "#c85131"),
    ]:
        r = np.linspace(source["radius"], 15, 301)
        profile = (r / source["radius"]) ** -source["q"] * (
            radial_basis(r, source["radius"], 15) @ cs
        )
        ax.plot(r, profile, label=name, color=color, lw=2)
    ax.set(
        xlabel="Emission radius (M)",
        ylabel="Emissivity / inner-edge emissivity",
        title="Frozen intrinsic radial emission · g³ excluded",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax = axes[1, 0]
    ax.scatter(length, abs(truth), s=3, color="#2d63ad", alpha=0.35, label="Truth")
    ax.scatter(length, abs(candidate), s=3, color="#c85131", alpha=0.35, label="Candidate")
    ax.set(
        xlabel="Measured baseline length (Gλ)",
        ylabel="Predicted |V| (Jy)",
        title="Predictions on all 6,568 retained baselines",
    )
    ax.legend(frameon=False, markerscale=3)
    ax = axes[1, 1]
    ax.scatter(length, residual.real, s=3, alpha=0.4, label="Real", color="#2d63ad")
    ax.scatter(length, residual.imag, s=3, alpha=0.4, label="Imaginary", color="#c85131")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(
        xlabel="Measured baseline length (Gλ)",
        ylabel="(Candidate − truth) / Isigma",
        title=f"Thermal-scaled residuals · total D = {np.linalg.norm(residual):.3f}",
    )
    ax.legend(frameon=False, markerscale=3)
    ax = axes[1, 2]
    fu = forecast["uv"] / 1e9
    for sign in (1, -1):
        points = ax.scatter(
            sign * fu[passed, 0],
            sign * fu[passed, 1],
            c=score[passed],
            s=26,
            cmap="viridis",
            vmin=0,
            vmax=score.max(),
            marker="s",
        )
        if np.any(~passed):
            ax.scatter(sign * fu[~passed, 0], sign * fu[~passed, 1], color="red", marker="x", s=22)
        ax.scatter(
            sign * uv[::20, 0] / 1e9, sign * uv[::20, 1] / 1e9, color="0.55", s=1, alpha=0.45
        )
    ax.set(
        xlabel="u (Gλ)",
        ylabel="v (Gλ)",
        aspect="equal",
        title="Forecast at checked nodes · assumed σ = 5 mJy",
    )
    fig.colorbar(points, ax=ax, shrink=0.8, label="|ΔV| / assumed σ per baseline")
    fig.suptitle(
        "Fixed traced geometries: observing nuisances and positive radial emission", fontsize=17
    )
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"pair_diagnostics.{extension}", dpi=170)
    plt.close(fig)
    rows = []
    for case in (
        "neutral",
        "legacy",
        "nuisance_only",
        "emission_only",
        "emission_fixed_nuisance",
        "joint",
    ):
        ds = summary["pair_distances"][case]
        rows.append(
            f"| {case} | {ds['n25']:.6f} | {ds['n35']:.6f} | {ds['n49']:.6f} | "
            f"{ds['jitter49']:.6f} | {ds['tight']:.6f} | {summary['accepted'][case]} |"
        )
    distance_table = "\n".join(rows)
    checks_table = "\n".join(
        f"| {name} | {comparison} | {d:.9f} |"
        for name in ("truth", label)
        for comparison, d in summary["comparisons"][name].items()
    )
    forecast_table = "\n".join(
        f"| {name} | {comparison} | {d:.9f} |"
        for name in ("truth", label)
        for comparison, d in summary["forecast_comparisons"][name].items()
    )
    d0 = summary["pair_distances"]["neutral"]["n49"]
    dn = summary["pair_distances"]["nuisance_only"]["n49"]
    dj = summary["pair_distances"]["joint"]["n49"]
    de = summary["pair_distances"]["emission_only"]["n49"]
    df = summary["pair_distances"]["emission_fixed_nuisance"]["n49"]
    dlegacy = summary["pair_distances"]["legacy"]["n49"]
    max_fourier = max(
        max(c[k] for k in ("truncation_D", "forecast_truncation_D"))
        for c in summary["checks"].values()
    )
    best = best_forecast[0]
    optimizer_status = "\n".join(
        f"- {name}: "
        + "; ".join(
            f"start {i}: D={record.get('distance', float('nan')):.6f}, {record['message']}"
            for i, record in enumerate(result["starts"])
        )
        for name, result in summary["fits"].items()
    )
    report = f"""# Radial emission and observing nuisances — 3 October 2026

The bounded study selected **{label}**,
with D={summary["pair_distances"][label]["n49"]:.6f} at regular49.
Measured-domain individual-source numerical checks pass: **{summary["numerical_pass"]}**.
A numerically checked D≤1 near-degenerate pair was found: **{summary["near_degenerate"]}**.
This target is operational, not a universal indistinguishability threshold
or detection significance.
A bounded failure does not establish uniqueness or rule out other geometries/emission families.

## What observing nuisances and emission absorb

At the two fixed traced geometries, using the existing q values and neutral observing parameters,
D={d0:.6f}. Fitting scale, rotation, position and flux alone reduces this to {dn:.6f}:
ΔD={d0 - dn:.6f}, removing {100 * (1 - dn**2 / d0**2):.3f}% of the initial D².
Adding the cubic radial family while holding those fitted nuisances fixed gives D={df:.6f}.
Jointly fitting emission and the same nuisance ranges gives D={dj:.6f}, an additional
ΔD={dn - dj:.6f} ({100 * (1 - dj**2 / dn**2):.3f}% of the nuisance-only residual D²).
Emission alone with neutral nuisances gives D={de:.6f}. These are conditional nested comparisons;
D reductions are not additive information components and depend on fitting order.

The old nuisance vector on regular49 subray predictions gives D={dlegacy:.6f}; the required
first refit gives {dn:.6f}. For the same regular49 source images represented as uniform pixels,
the old-vector pair gives D={representation["uniform_legacy_pair_D"]:.6f}. Individual-source
subray-to-uniform differences are {representation["truth"]["subray_to_uniform_D"]:.6f} / 
{representation["legacy"]["subray_to_uniform_D"]:.6f}. These are representation differences,
not numerical convergence estimates. No observed amplitudes or phases were fitted.

| Frozen case | regular25 D | regular35 D | regular49 D | jitter49 D | tight25 D | Source pass |
|---|---:|---:|---:|---:|---:|---|
{distance_table}

The final column concerns the individual candidate's four sampling/tolerance comparisons;
truth must also pass. All fitted parameters were written to `frozen_candidates.json` before
these checks. No setting-specific refit was performed. Optimizer outcomes and every start's
objective are in `summary.json`; a local optimum is not a global search certificate.
Evaluation-limit stops are incomplete optimization, even if numerical sampling checks pass.

{optimizer_status}

![Images, radial emission, measured predictions/residuals and uv forecast](pair_diagnostics.png)

## Frozen inputs and restrictions

Protocol: `{protocol["created_utc"]}`; source commit `{protocol["consumer_commit"]}`.
Retained renderer SHA-256 `{protocol["binary"]["binary_sha256"]}` verified against adjacent
identity.json and the prior protocol. Neither renderer source nor retained builds was modified.
Truth: a=0.98, i=18°, [r_in,r_out]=[8,15] M, q=2, g_power=3, 0.6 Jy, 125 μas.
Candidate geometry: a=0, i={sources["candidate"]["inclination"]:.10f}°,
r_in={sources["candidate"]["radius"]:.10f} M, r_out=15 M.

The nested family is I=g³(r/r_in)^(-q) Σ c_k B_k³(t),
t=log(r/r_in)/log(r_out/r_in). q stays at {sources["candidate"]["q"]:.10f}; c0=1,
c1..3 ∈ [0.25,4]. All c=1 exactly recover the existing power law. This compares a fixed
existing exponent to a three-parameter modifier; it is not a free-q optimization. The modifier
is positive and cubic, with no internal knots, |d modifier/dt|≤11.25 and
|d² modifier/dt²|≤45. It can change radial weighting smoothly but cannot move or eliminate
an emitting edge, expose later crossings or create arbitrary narrow emission features.

Selected coefficients: `{fit["coefficients"]}`.
Selected nuisance vector [scale, angle degrees, east μas, north μas, flux Jy]:
`{fit["nuisance"]}`. Bounds are [0.95,1.05], [-10,10]°, [-3,3] μas for each position,
[0.3,1.2] Jy; identical wherever nuisances are varied. Ideal calibration; no station gains.
Three fixed starts per fit, max55 function evaluations per start, TRF least squares,
ftol=xtol=gtol=1e-8. Exact basis moments avoid emission interpolation and retracing.

The camera is at 85M with 22° horizontal FOV, assigned 125 μas times fitted scale on a flat
screen. This is not an observer-at-infinity, exact-solid-angle or calibrated mass/distance
construction. Original first-hit masks, all samples, failures and weights are retained.
ISCO clamping is unchanged; these edges are outside both spins' ISCO, so the clamp is inactive
for these specific geometries. This study cannot extrapolate to disks reaching their ISCO or
plunging flows. Emission-family restrictions and the retained opaque thin-disk model limit
any inference about spin. No training or reserved-test use occurred.

## Numerical evidence for the selected pair

D²=sum |ΔV|²/Isigma² on all 6,568 retained ≥0.1 Gλ baselines, no averaging.
Every source must have D<0.25 in two successive refinements, a distinct pattern comparison,
and the reused same-node tolerance sensitivity. The old failed 15→25 records remain unchanged.

| Source | Frozen-source comparison | D |
|---|---|---:|
{checks_table}

The jitter pattern is deterministic and repeated across pixels; it is not an independent
random ensemble and gives no Monte Carlo uncertainty. The tight25 check varies integrator
tolerance from 1e-9 to1e-11, not intersection/root algorithms. These are selected-case checks,
not continuum convergence or physical-model validation. All ten cache exports retain valid
sample quality, no unfinished rays or missing hit redshifts, and exact stored-radiance pixel
reconstruction. The changed radial weighting was re-evaluated at every setting.

The maximum norm of the Fourier truncation bounds over all measured/forecast settings is
{max_fourier:.4g}. Each selected candidate's full 6,568-baseline regular49 direct quadrature
plus all {summary["forecast_nodes"]} forecast nodes was compared with the accelerated sum.
Direct comparison records: `{summary["full_direct_checks"]}`. Other frozen cases/settings
include16 spread measured-baseline direct checks; truth has32 measured plus the full forecast.
All direct-error and bound gates are <1e-7; zero flux/conjugacy gate is2e-13 Jy.
The prior truth's full measured direct comparison is also retained in the refinement run.

## What remains distinguishable, and a forecast

The measured residual has real-component D²={diagnostic["residual_real_D2"]:.6f} and
imaginary-component D²={diagnostic["residual_imag_D2"]:.6f}. `diagnostics.json` gives the
baseline-length contributions and largest residual coordinates, preserving complex residuals
rather than using amplitude alone. The sources have different transfer/redshift patterns;
these residuals cannot be assigned uniquely to a physical feature by this radial-only study.

The fixed forecast uses {summary["forecast_nodes"]} independent half-plane nodes on a1 Gλ
Cartesian grid inside12 Gλ; plotted conjugates are not extra independent measurements.
Assumed thermal standard deviation is **5 mJy per real/imaginary component**, ideal calibration.
This is a forecast for this frozen pair, not a proposed observing campaign or actual sensitivity.
Whole-domain sampling norm pass: **{summary["forecast_domain_pass"]}**;
pointwise passing nodes: **{summary["forecast_pointwise_passing"]}/{summary["forecast_nodes"]}**.
Fourier approximation is checked at every forecast node by direct quadrature and bounded at
all settings. No accuracy claim is made between these nodes or beyond this domain.

| Source | Forecast-domain comparison | D at assumed sensitivity |
|---|---|---:|
{forecast_table}

The highest passing forecast-node separation is at (u,v)={best["uv_glambda"]} Gλ:
|ΔV|={best["difference_mjy"]:.4f} mJy, single-baseline D={best["single_baseline_D"]:.4f} at5 mJy,
maximum per-source sampling change {best["maximum_sampling_change_in_sigma"]:.5f}σ.
The conjugate coordinate has the same separation, not independent spin information.
This identifies where added directional coverage would best separate this particular pair
under the stated sensitivity. Source variability, scattering and instrumental gain errors
are outside this study. If the aggregate forecast sampling gate fails, only explicitly
passing nodes carry the pointwise numerical statement; the full-domain forecast is unresolved.

## Budget, preservation and reproduction

New renders: **0**, within the authorized12. Driver wall time {summary["wall_seconds"]:.2f}s,
artifacts at driver completion {summary["artifact_bytes"]:,}bytes. Tests/report/QA overhead is
recorded in `verification.json`; the authorized ceilings are2700s computation and6GB additional
artifacts. No geometry search was attempted: both retained geometries already supply the full
refinement ladder and isolate nuisance/emission effects. No post-result extension occurred.
Protected historical pilot, EHT, refinement and retained tool files unchanged:
**{summary["historical_artifacts_unchanged"]}** ({len(protocol["protected_files"])} hashes).

```sh
.venv/bin/python scripts/fit_transfer_emission.py --run transfer_emission_FRESH
MPLCONFIGDIR=/tmp/kerr-mpl .venv/bin/python scripts/report_transfer_emission.py \\
  --run transfer_emission_FRESH
MPLCONFIGDIR=/tmp/kerr-mpl .venv/bin/pytest -q -m 'not slow'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Run artifacts include immutable protocol and candidate freezes; cache references and hashes;
source/config snapshots; all basis moments, complex predictions, direct sums and quality checks;
figures and diagnostics. Reproduction requires the original retained caches and renderer at
the recorded identities. Do not rerun into this completed run directory.
"""
    (folder / "RESULTS.md").write_text(report)
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
