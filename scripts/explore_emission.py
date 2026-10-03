import argparse
import itertools
import json
import shutil
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from kerr_sbi.config import load_config
from kerr_sbi.obs_model import luminance
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.pfm import read_pfm
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.simulator import render
from kerr_sbi.visibility import (
    FourierImage,
    direct_visibility,
    load_eht_csv,
    matched_flux,
    visibility_distance,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search emission nuisances for a spin ambiguity.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/visibility_emission.toml")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("run must be a single directory name")
    settings = tomllib.loads(args.settings.read_text())["experiment"]
    axes = [
        np.array(settings[key])
        for key in ("radius_nodes", "inclination_nodes_deg", "emissivity_nodes")
    ]
    if any(len(axis) < 2 or np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError("grid axes must increase")
    if np.prod([len(axis) for axis in axes]) + 4 > settings["new_render_budget"]:
        raise ValueError("grid and confirmations exceed render budget")
    destination = ROOT / "runs" / args.run
    destination.mkdir(exist_ok=False)
    shutil.copyfile(__file__, destination / "explore_emission.py")
    shutil.copyfile(args.settings, destination / "settings.toml")
    baseline = ROOT / "runs" / settings["baseline_run"]
    data = load_eht_csv(baseline / "observations.csv", settings["minimum_baseline_lambda"])
    cfg = load_config()
    cfg["scene"]["supersample"] = settings["supersample"]
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "settings": settings,
        "nullgeo": simulator_provenance(cfg),
        "csv_sha256": file_sha256(baseline / "observations.csv"),
        "baseline_protocol_sha256": file_sha256(baseline / "protocol.json"),
        "source_hashes": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [Path(__file__), ROOT / "src/kerr_sbi/visibility.py", args.settings]
        },
        "interpretation": [
            "Adaptive follow-up selected from the baseline's closest case; development only.",
            "Synthetic means, same ideal thermal experiment; no fit to observed source values.",
            "Trilinear normalized-image interpolation proposes candidates; direct renders decide.",
            "Finite bounded grid and local search cannot establish global uniqueness.",
            "15x15 renders, selected pair checked at 25x25; checks are not continuum validation.",
            "No model training and no reserved test split use.",
        ],
    }
    write_json(destination / "protocol.json", protocol)
    render_records = []

    def image_for(spin: float, point: np.ndarray, name: str, sampling: int) -> np.ndarray:
        if len(render_records) >= settings["new_render_budget"]:
            raise RuntimeError("render budget exhausted")
        changed = deepcopy(cfg)
        radius, inclination, emissivity = map(float, point)
        changed["scene"]["r_in"] = radius
        changed["scene"]["supersample"] = sampling
        changed["emission"]["emissivity_index"] = emissivity
        path = render(spin, inclination, destination / "renders" / name, changed)
        render_records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "spin": spin,
                "radius": radius,
                "inclination": inclination,
                "emissivity_index": emissivity,
                "supersample": sampling,
            }
        )
        write_json(destination / "renders.json", render_records)
        print(f"Render {len(render_records)}/{settings['new_render_budget']}: {name}", flush=True)
        return luminance(read_pfm(path))

    models = {}
    for index in itertools.product(*(range(len(axis)) for axis in axes)):
        point = np.array([axis[j] for axis, j in zip(axes, index, strict=True)])
        image = image_for(
            settings["alternative_spin"],
            point,
            "grid_" + "_".join(map(str, index)),
            settings["supersample"],
        )
        models[index] = FourierImage(image, settings["fft_padding"])
    pixel = settings["fov_uas"] / 64
    truth_point = np.array(
        [
            settings["truth_inner_radius"],
            settings["truth_inclination_deg"],
            settings["truth_emissivity_index"],
        ]
    )
    truth_image = image_for(settings["truth_spin"], truth_point, "truth", settings["supersample"])
    truth = settings["flux_jy"] * direct_visibility(truth_image, data.uv, pixel)

    def predict(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        point, scale, angle, dx, dy = parameters[:3], *parameters[3:]
        left = [
            int(np.clip(np.searchsorted(axis, value) - 1, 0, len(axis) - 2))
            for axis, value in zip(axes, point, strict=True)
        ]
        fractions = [
            (value - axis[j]) / (axis[j + 1] - axis[j])
            for axis, value, j in zip(axes, point, left, strict=True)
        ]
        prediction = np.zeros(len(data.sigma), dtype=complex)
        for offset in itertools.product((0, 1), repeat=3):
            index = tuple(j + k for j, k in zip(left, offset, strict=True))
            weight = np.prod(
                [
                    fraction if k else 1 - fraction
                    for fraction, k in zip(fractions, offset, strict=True)
                ]
            )
            if weight:
                prediction += weight * models[index].evaluate(
                    data.uv, pixel * scale, angle, (dx, dy)
                )
        flux = matched_flux(prediction, truth, data.sigma, tuple(settings["flux_bounds_jy"]))
        return flux * prediction, flux

    def residual(parameters: np.ndarray) -> np.ndarray:
        difference = (predict(parameters)[0] - truth) / data.sigma
        return np.concatenate((difference.real, difference.imag))

    bounds = np.array([settings["parameter_lower"], settings["parameter_upper"]])
    fits = []
    for start in settings["starts"]:
        fit = least_squares(
            residual,
            start,
            bounds=bounds,
            x_scale=[0.2, 2, 0.2, 0.05, 10, 3, 3],
            max_nfev=settings["max_evaluations"],
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        fits.append(fit)
        print(f"Fit D={np.linalg.norm(fit.fun):.5f}, parameters={fit.x}", flush=True)
    best = min(fits, key=lambda fit: np.linalg.norm(fit.fun))
    approximate, flux = predict(best.x)
    point, scale, angle, dx, dy = best.x[:3], *best.x[3:]
    selected = image_for(settings["alternative_spin"], point, "selected", settings["supersample"])
    direct = flux * direct_visibility(selected, data.uv, pixel * scale, angle, (dx, dy))
    truth_confirmed_image = image_for(
        settings["truth_spin"],
        truth_point,
        "truth_confirmation",
        settings["confirmation_supersample"],
    )
    selected_confirmed_image = image_for(
        settings["alternative_spin"],
        point,
        "selected_confirmation",
        settings["confirmation_supersample"],
    )
    truth_confirmed = settings["flux_jy"] * direct_visibility(truth_confirmed_image, data.uv, pixel)
    selected_confirmed = flux * direct_visibility(
        selected_confirmed_image, data.uv, pixel * scale, angle, (dx, dy)
    )
    interpolation = visibility_distance(direct, approximate, data.sigma)
    errors = [
        visibility_distance(truth, truth_confirmed, data.sigma),
        visibility_distance(direct, selected_confirmed, data.sigma),
    ]
    confirmed_distance = visibility_distance(truth_confirmed, selected_confirmed, data.sigma)
    result = {
        "parameters_order": [
            "r_in",
            "inclination_deg",
            "emissivity_index",
            "scale",
            "angle_deg",
            "east_offset_uas",
            "north_offset_uas",
        ],
        "parameters": best.x.tolist(),
        "flux_jy": flux,
        "interpolated_distance": float(np.linalg.norm(best.fun)),
        "direct_distance": visibility_distance(truth, direct, data.sigma),
        "confirmation_distance": confirmed_distance,
        "simple_pair_error_probability": float(ndtr(-confirmed_distance / 2)),
        "interpolation_noise_error": interpolation,
        "sampling_noise_errors": errors,
        "interpolation_passed": interpolation < settings["max_interpolation_noise_norm"],
        "sampling_passed": max(errors) < settings["max_confirmation_noise_norm"],
        "optimizer_success": bool(best.success),
        "optimizer_message": best.message,
        "near_bound": bool(
            np.any(
                np.minimum(best.x - bounds[0], bounds[1] - best.x) < 1e-4 * (bounds[1] - bounds[0])
            )
        ),
        "render_count": len(render_records),
    }
    write_json(destination / "summary.json", result)
    np.savez_compressed(
        destination / "predictions.npz",
        uv=data.uv,
        sigma=data.sigma,
        truth=truth,
        selected=direct,
        truth_confirmed=truth_confirmed,
        selected_confirmed=selected_confirmed,
        truth_image=truth_confirmed_image,
        selected_image=selected_confirmed_image,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
