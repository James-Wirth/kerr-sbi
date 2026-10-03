import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from explore_visibility import ROOT, cached_image

from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.visibility import (
    direct_visibility,
    fit_station_gains,
    load_eht_csv,
    visibility_distance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check EHT forecast sampling and station gains.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if any(Path(value).name != value for value in (args.source, args.run)):
        raise ValueError("run names must be single directory names")
    source, destination = ROOT / "runs" / args.source, ROOT / "runs" / args.run
    destination.mkdir(exist_ok=False)
    shutil.copyfile(__file__, destination / "diagnose_visibility.py")
    shutil.copyfile(ROOT / "src/kerr_sbi/visibility.py", destination / "visibility_source.py")
    baseline_protocol = json.loads((source / "protocol.json").read_text())
    cases = json.loads((source / "summary.json").read_text())["cases"]
    settings = baseline_protocol["settings"]
    data = load_eht_csv(source / "observations.csv", settings["minimum_baseline_lambda"])
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_protocol_sha256": file_sha256(source / "protocol.json"),
        "source_summary_sha256": file_sha256(source / "summary.json"),
        "source_hashes": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [
                Path(__file__),
                ROOT / "src/kerr_sbi/visibility.py",
                ROOT / "scripts/explore_visibility.py",
            ]
        },
        "gain_intervals_seconds": [120, 300],
        "gain_amplitude_bounds": [0.8, 1.2],
        "interpretation": [
            "Fixed fitted geometry from baseline; no refit at 15x15.",
            "Gains independent in UTC-aligned intervals; phase reference fixed per group.",
            "Gain amplitudes bounded to [0.8,1.2], phases free; no calibration priors.",
            "Sensitivity scenario, not an empirical EHT calibration model or actual data fit.",
            "D after profiling gains is a fitted distance, not a detection significance.",
        ],
        "new_renders": 0,
    }
    write_json(destination / "protocol.json", protocol)
    pixel = settings["fov_uas"] / 64
    results, arrays = [], {"uv": data.uv, "sigma": data.sigma}
    for case in cases:
        if case["family"] != "fixed":
            continue
        fit, inclination = case["best"], case["inclination_deg"]
        scale, angle, dx, dy = fit["parameters"]
        predictions, records = [], []
        for sampling, directory in [(9, "fixed_radius_renders"), (15, "fixed_radius_ss15")]:
            truth_image, truth_record = cached_image(
                ROOT / settings["cache"] / directory, settings["truth_spin"], inclination
            )
            image, record = cached_image(
                ROOT / settings["cache"] / directory, settings["alternative_spin"], inclination
            )
            truth = settings["flux_jy"] * direct_visibility(truth_image, data.uv, pixel)
            alternative = fit["flux_jy"] * direct_visibility(
                image, data.uv, pixel * scale, angle, (dx, dy)
            )
            predictions.append((truth, alternative))
            records.append({"sampling": sampling, "truth": truth_record, "alternative": record})
        gains = []
        for interval in protocol["gain_intervals_seconds"]:
            adjusted, result = fit_station_gains(
                *predictions[1], data, interval, tuple(protocol["gain_amplitude_bounds"])
            )
            gains.append(result)
            arrays[f"i{inclination}_gains{interval}"] = adjusted
        result = {
            "inclination_deg": inclination,
            "distance_9": visibility_distance(*predictions[0], data.sigma),
            "distance_15": visibility_distance(*predictions[1], data.sigma),
            "sampling_errors": [
                visibility_distance(predictions[0][j], predictions[1][j], data.sigma)
                for j in (0, 1)
            ],
            "gain_fits": gains,
            "renders": records,
        }
        results.append(result)
        arrays[f"i{inclination}_truth"], arrays[f"i{inclination}_alternative"] = predictions[1]
        print(
            f"i={inclination}: D15={result['distance_15']:.3f}, "
            f"gain-fitted D={[round(gain['distance'], 3) for gain in gains]}",
            flush=True,
        )
    write_json(destination / "summary.json", {"cases": results})
    np.savez_compressed(destination / "predictions.npz", **arrays)


if __name__ == "__main__":
    main()
