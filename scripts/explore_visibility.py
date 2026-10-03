import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from kerr_sbi.obs_model import luminance
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.pfm import read_pfm
from kerr_sbi.visibility import (
    FourierImage,
    direct_visibility,
    fit_visibility_pair,
    load_eht_csv,
    visibility_distance,
)

ROOT = Path(__file__).resolve().parents[1]


def cached_image(directory: Path, spin: float, inclination: float) -> tuple[np.ndarray, dict]:
    parameters = [float(f"{spin:.15g}"), float(f"{inclination:.15g}")]
    key = hashlib.sha256(json.dumps(parameters).encode()).hexdigest()[:24]
    record_path, path = directory / f"{key}.json", directory / f"{key}.pfm"
    record = json.loads(record_path.read_text())
    if record["parameters"] != parameters or file_sha256(path) != record["sha256"]:
        raise ValueError(f"cached render failed integrity check: {path}")
    return luminance(read_pfm(path)), {
        "path": str(path.relative_to(ROOT)),
        "sha256": record["sha256"],
        "metadata_sha256": file_sha256(record_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore spin ambiguity with public EHT sampling.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/visibility.toml")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("run must be a single directory name")
    settings = tomllib.loads(args.settings.read_text())["experiment"]
    if settings["new_render_budget"] != 0 or settings["training_budget"] != 0:
        raise ValueError("this cached-render experiment does not render or train")
    destination = ROOT / "runs" / args.run
    destination.mkdir(exist_ok=False)
    shutil.copyfile(args.csv, destination / "observations.csv")
    shutil.copyfile(args.settings, destination / "settings.toml")
    shutil.copyfile(__file__, destination / "explore_visibility.py")
    data = load_eht_csv(args.csv, settings["minimum_baseline_lambda"])
    cache = ROOT / settings["cache"]
    directories = {"isco": cache / "renders", "fixed": cache / "fixed_radius_renders"}
    images, provenance, identities = {}, {}, {}
    for family in settings["families"]:
        identity_path = directories[family] / "identity.json"
        identity = json.loads(identity_path.read_text())
        identities[family] = {"sha256": file_sha256(identity_path), "identity": identity}
        scene, emission = identity["config"]["scene"], identity["config"]["emission"]
        if scene["width"] != 64 or scene["height"] != 64 or scene["supersample"] != 9:
            raise ValueError("unexpected cached scene resolution")
        if emission != {"model": "stylized", "g_power": 3.0, "emissivity_index": 2.0}:
            raise ValueError("unexpected cached emission model")
        if scene["r_in"] != (0 if family == "isco" else 8):
            raise ValueError("unexpected cached inner radius")
        for inclination in settings["inclinations_deg"]:
            for spin in (settings["truth_spin"], settings["alternative_spin"]):
                key = f"{family}/{inclination}/{spin}"
                images[key], provenance[key] = cached_image(directories[family], spin, inclination)
    sources = [
        Path(__file__),
        ROOT / "src/kerr_sbi/visibility.py",
        ROOT / "src/kerr_sbi/pfm.py",
        ROOT / "src/kerr_sbi/obs_model.py",
        ROOT / "uv.lock",
        args.settings,
    ]
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "settings": settings,
        "source_hashes": {str(path.relative_to(ROOT)): file_sha256(path) for path in sources},
        "input_sha256": file_sha256(args.csv),
        "data_source": "https://github.com/eventhorizontelescope/2019-D01-01",
        "data_header": data.header,
        "measurements_used": len(data.sigma),
        "cache_identities": identities,
        "renders": provenance,
        "question": "Do spin ambiguities survive EHT sampling under ideal thermal noise?",
        "interpretation": [
            "Synthetic mean visibilities with real EHT uv coordinates and reported thermal sigma.",
            "Observed amplitudes and phases are not fitted or used to select models.",
            "Independent real and imaginary noise each have standard deviation Isigma.",
            "D is full-data noise distance; D^2/2 is Gaussian KL for a fixed pair.",
            "Equal-prior simple-hypothesis error is Phi(-D/2), not a composite-model p-value.",
            "Known station calibration and fixed source during the night: optimistic information.",
            "Inclination fixed within pairs; radius family changes in the discrete comparison.",
            "Uniform square pixels, top-down image y maps to north, x maps to east.",
            "Positive orientation rotates east toward north; not astronomical position angle.",
            "125 microarcsec image FOV is an illustrative choice, not fitted mass-to-distance.",
            "Observer at 85M; this is not an asymptotic-camera or realistic plasma model.",
            "Multistart local optimization finds examples, not a global identifiability bound.",
            "Direct sums verify every best pair; numerical threshold fixed before evaluation.",
        ],
    }
    write_json(destination / "protocol.json", protocol)
    pixel = settings["fov_uas"] / 64
    bounds = np.array(
        [
            settings["scale_bounds"],
            settings["angle_bounds_deg"],
            settings["offset_bounds_uas"],
            settings["offset_bounds_uas"],
        ]
    ).T
    results, arrays = [], {"uv": data.uv, "sigma": data.sigma}
    for family in settings["families"]:
        for inclination in settings["inclinations_deg"]:
            key = f"{family}/{inclination}/{settings['truth_spin']}"
            truth_image = images[key]
            truth = settings["flux_jy"] * direct_visibility(
                truth_image, data.uv, pixel, settings["orientation_deg"]
            )
            truth_fft = settings["flux_jy"] * FourierImage(
                truth_image, settings["fft_padding"]
            ).evaluate(data.uv, pixel, settings["orientation_deg"])
            truth_error = visibility_distance(truth, truth_fft, data.sigma)
            alternatives = []
            for alternative_family in settings["families"]:
                alternative = images[
                    f"{alternative_family}/{inclination}/{settings['alternative_spin']}"
                ]
                model = FourierImage(alternative, settings["fft_padding"])
                known = settings["flux_jy"] * direct_visibility(
                    alternative, data.uv, pixel, settings["orientation_deg"]
                )
                fit = fit_visibility_pair(
                    model,
                    truth,
                    data.uv,
                    data.sigma,
                    pixel,
                    bounds,
                    tuple(settings["flux_bounds_jy"]),
                    settings["starts"],
                    settings["max_evaluations"],
                )
                scale, angle, dx, dy = fit["parameters"]
                direct = fit["flux_jy"] * direct_visibility(
                    alternative, data.uv, pixel * scale, angle, (dx, dy)
                )
                approximate = fit["flux_jy"] * model.evaluate(
                    data.uv, pixel * scale, angle, (dx, dy)
                )
                error = visibility_distance(direct, approximate, data.sigma)
                distance = visibility_distance(truth, direct, data.sigma)
                fit.update(
                    {
                        "alternative_family": alternative_family,
                        "known_nuisance_distance": visibility_distance(truth, known, data.sigma),
                        "direct_distance": distance,
                        "fft_noise_error": error,
                        "truth_fft_noise_error": truth_error,
                        "fourier_check_passed": max(error, truth_error)
                        < settings["max_fft_noise_norm"],
                        "simple_pair_error_probability": float(ndtr(-distance / 2)),
                    }
                )
                alternatives.append(fit)
                arrays[f"case_{len(results)}_{alternative_family}"] = direct
                print(
                    f"{family} i={inclination:g}, alternative={alternative_family}: "
                    f"D fixed={fit['known_nuisance_distance']:.3f}, fitted={distance:.3f}, "
                    f"Fourier error={error:.5f}",
                    flush=True,
                )
            best = min(alternatives, key=lambda fit: fit["direct_distance"])
            results.append(
                {
                    "family": family,
                    "inclination_deg": inclination,
                    "alternatives": alternatives,
                    "best": best,
                }
            )
            arrays[f"case_{len(results) - 1}_truth"] = truth
            write_json(destination / "status.json", {"completed_cases": len(results)})
    summary = {
        "cases": results,
        "all_fourier_checks_passed": all(
            fit["fourier_check_passed"] for case in results for fit in case["alternatives"]
        ),
    }
    write_json(destination / "summary.json", summary)
    np.savez_compressed(destination / "predictions.npz", **arrays)
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for panel, family in zip(panels, settings["families"], strict=True):
        cases = [case for case in results if case["family"] == family]
        same = [
            next(fit for fit in case["alternatives"] if fit["alternative_family"] == family)
            for case in cases
        ]
        x = [case["inclination_deg"] for case in cases]
        panel.plot(
            x,
            [fit["known_nuisance_distance"] for fit in same],
            "o-",
            label="Geometry and flux fixed",
        )
        panel.plot(
            x,
            [fit["direct_distance"] for fit in same],
            "s-",
            label="Fit scale, rotation, position, flux",
        )
        panel.plot(
            x,
            [case["best"]["direct_distance"] for case in cases],
            "^-",
            label="Also choose disk-edge family",
        )
        panel.axhline(2, color="gray", ls=":", label="D=2: fixed-pair error 16%")
        panel.set(
            xlabel="Known inclination (degrees)",
            ylabel="Full-data thermal distance D",
            title=f"Truth: {'ISCO edge' if family == 'isco' else 'inner edge 8M'}, spin 0.98 vs 0",
            yscale="log",
        )
    panels[0].legend(fontsize=8)
    fig.suptitle("Synthetic sources • EHT 2017 Apr 11 sampling • ideal calibration")
    fig.savefig(destination / "spin_separation.png", dpi=170)
    plt.close(fig)
    index = min(range(len(results)), key=lambda j: results[j]["best"]["direct_distance"])
    selected = results[index]
    target, alternative = (
        arrays[f"case_{index}_truth"],
        arrays[f"case_{index}_{selected['best']['alternative_family']}"],
    )
    baseline = np.linalg.norm(data.uv, axis=1) / 1e9
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    panels[0].scatter(baseline, np.abs(target), s=2, label="Spin 0.98 truth")
    panels[0].scatter(baseline, np.abs(alternative), s=2, label="Spin 0 alternative")
    panels[0].set(
        xlabel="Baseline length (Gλ)",
        ylabel="Synthetic visibility amplitude (Jy)",
        title="Closest fitted pair",
    )
    panels[0].legend()
    panels[1].scatter(baseline, np.abs(target - alternative) / data.sigma, s=2)
    panels[1].set(
        xlabel="Baseline length (Gλ)",
        ylabel="Complex difference / per-component thermal σ",
        title=f"All measurements combined: D={selected['best']['direct_distance']:.2f}",
    )
    fig.savefig(destination / "closest_pair.png", dpi=170)
    plt.close(fig)
    print(f"Saved {destination}; Fourier checks: {summary['all_fourier_checks_passed']}")


if __name__ == "__main__":
    main()
