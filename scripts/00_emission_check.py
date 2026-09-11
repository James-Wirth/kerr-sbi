import argparse
import csv
import hashlib
import json
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from kerr_sbi.config import DEFAULT_CONFIG, Config, load_config, project_path
from kerr_sbi.obs_model import luminance
from kerr_sbi.pfm import read_pfm
from kerr_sbi.provenance import simulator_provenance
from kerr_sbi.simulator import build_scene, render, template_sha256


def measure_emission(rgb: np.ndarray) -> dict[str, Any]:
    y = luminance(rgb)
    total_rgb = rgb.sum(axis=-1, dtype=np.float64)
    disk = total_rgb > 0
    if not np.any(disk):
        raise ValueError("render contains no positive disk radiance")
    red_fraction = rgb[..., 0][disk].astype(np.float64) / total_rgb[disk]
    middle = y.shape[1] // 2
    left_flux = float(y[:, :middle].sum(dtype=np.float64))
    right_flux = float(y[:, middle:].sum(dtype=np.float64))
    if right_flux <= 0:
        raise ValueError("render has no right-half flux")
    return {
        "peak_luminance": float(y.max()),
        "left_flux": left_flux,
        "right_flux": right_flux,
        "left_right_flux_ratio": left_flux / right_flux,
        "red_fraction_min": float(red_fraction.min()),
        "red_fraction_max": float(red_fraction.max()),
        "red_fraction_spread": float(np.ptp(red_fraction)),
        "rgb_channels_identical": bool(
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
        ),
        "vertical_flux_profile": (
            y.sum(axis=1, dtype=np.float64) / y.sum(dtype=np.float64)
        ).tolist(),
    }


def save_montage(
    cases: list[dict[str, Any]], label: str, model: str, cfg: Config, directory: Path
) -> None:
    check = cfg["emission_check"]
    rows, columns = len(check["spins"]), len(check["inclinations_deg"])
    fig, axes = plt.subplots(rows, columns, figsize=(10, 10.5), squeeze=False)
    for ax, case in zip(axes.flat, cases, strict=True):
        rgb = read_pfm(Path(cfg["project_root"]) / case["pfm_path"])
        peak = float(rgb.max())
        scale = check["asinh_peak_fraction"] * peak
        display = np.arcsinh(rgb / scale) / np.arcsinh(peak / scale)
        ax.imshow(np.clip(display, 0, 1), origin="upper", interpolation="nearest")
        ax.set_title(f"a = {case['a']:g}     i = {case['incl_deg']:g}°", fontsize=12)
        ax.set_xlabel(
            f"peak Y = {case['peak_luminance']:.4g}     L/R = {case['left_right_flux_ratio']:.2f}×",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    title = (
        f"{label} · Stylized: g power = {cfg['emission']['g_power']:g}, "
        f"emissivity index = {cfg['emission']['emissivity_index']:g}"
        if model == "stylized"
        else f"{label} · Blackbody: T in = {cfg['emission']['t_in']:,.0f} K"
    )
    fig.suptitle(title, fontsize=17, y=0.985)
    fig.text(
        0.5,
        0.948,
        "Linear RGB · per-panel asinh stretch and peak normalization · 64×64 renders",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.905, bottom=0.045, wspace=0.12, hspace=0.28)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"m1_{label}.{suffix}", dpi=check["figure_dpi"])
    plt.close(fig)


def save_edge_profiles(options: dict[str, Any], cfg: Config, directory: Path) -> None:
    check = cfg["emission_check"]
    inclination = max(check["inclinations_deg"])
    fig, axes = plt.subplots(1, len(check["spins"]), figsize=(12, 3.6), squeeze=False)
    row_offsets = np.arange(cfg["scene"]["height"]) - (cfg["scene"]["height"] - 1) / 2
    for ax, a in zip(axes.flat, check["spins"], strict=True):
        for label, color in (("A", "#265a9c"), ("B", "#c26430")):
            case = next(
                case
                for case in options[label]["cases"]
                if case["a"] == a and case["incl_deg"] == inclination
            )
            ax.plot(row_offsets, case["vertical_flux_profile"], label=label, color=color)
        ax.set_title(f"a = {a:g}, i = {inclination:g}°")
        ax.set_xlabel("Image row offset (px; positive downward)")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Fraction of total flux per row")
    axes[0, -1].legend(title="A: stylized\nB: blackbody")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"m1_edge_profiles.{suffix}", dpi=check["figure_dpi"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    check = cfg["emission_check"]
    provenance = simulator_provenance(cfg)
    if check["asinh_peak_fraction"] <= 0:
        raise ValueError("asinh_peak_fraction must be positive")
    data_directory = project_path(cfg, "data") / "m1"
    figures = project_path(cfg, "figures")
    results = project_path(cfg, "results")
    for directory in (data_directory, figures, results):
        directory.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {}
    total = 2 * len(check["spins"]) * len(check["inclinations_deg"])
    with tqdm(total=total, desc="M1 sequential renders", unit="image") as progress:
        for label, model in (("A", "stylized"), ("B", "blackbody")):
            case_cfg = deepcopy(cfg)
            case_cfg["emission"]["model"] = model
            case_cfg["simulator"]["smoke_png"] = False
            cases = []
            for a in check["spins"]:
                for inclination in check["inclinations_deg"]:
                    stem = data_directory / f"{label}_a{a:g}_i{inclination:g}"
                    started = time.perf_counter()
                    path = render(a, inclination, stem, case_cfg)
                    elapsed = time.perf_counter() - started
                    rgb = read_pfm(path)
                    if rgb.shape != (cfg["scene"]["height"], cfg["scene"]["width"], 3):
                        raise ValueError(f"unexpected render shape: {rgb.shape}")
                    scene = build_scene(a, inclination, path, case_cfg)
                    path.with_suffix(".toml").write_text(scene)
                    cases.append(
                        {
                            "option": label,
                            "model": model,
                            "a": a,
                            "incl_deg": inclination,
                            "pfm_path": str(path.relative_to(cfg["project_root"])),
                            "pfm_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "scene_sha256": hashlib.sha256(scene.encode()).hexdigest(),
                            "render_seconds": elapsed,
                            **measure_emission(rgb),
                        }
                    )
                    progress.update()
            peaks = [case["peak_luminance"] for case in cases]
            red_spread = max(case["red_fraction_max"] for case in cases) - min(
                case["red_fraction_min"] for case in cases
            )
            options[label] = {
                "model": model,
                "peak_luminance_table": np.array(peaks)
                .reshape(len(check["spins"]), len(check["inclinations_deg"]))
                .tolist(),
                "peak_max_min_ratio": max(peaks) / min(peaks),
                "edge_on_left_right_ratios": [
                    case["left_right_flux_ratio"]
                    for case in cases
                    if case["incl_deg"] == max(check["inclinations_deg"])
                ],
                "red_fraction_spread": red_spread,
                "achromatic": red_spread <= check["achromatic_tolerance"],
                "rgb_channels_identical": all(case["rgb_channels_identical"] for case in cases),
                "cases": cases,
            }
            save_montage(cases, label, model, cfg, figures)
    save_edge_profiles(options, cfg, figures)
    report = {
        "date_utc": datetime.now(UTC).isoformat(),
        "nullgeo": provenance,
        "template_sha256": template_sha256(cfg),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scene": cfg["scene"],
        "emission_parameters": cfg["emission"],
        "measurement_space": "raw linear PFM; before PSF and flux normalization",
        "grid_rows_a": check["spins"],
        "grid_columns_incl_deg": check["inclinations_deg"],
        "check_settings": check,
        "options": options,
        "stylized_meets_numeric_decision_rule": bool(
            options["A"]["achromatic"]
            and options["A"]["peak_max_min_ratio"] <= check["peak_range_limit"]
        ),
        "selection_status": "awaiting user choice at M1 gate",
    }
    (results / "m1_emission_check.json").write_text(json.dumps(report, indent=2) + "\n")
    table_rows = [
        {key: value for key, value in case.items() if key != "vertical_flux_profile"}
        for option in options.values()
        for case in option["cases"]
    ]
    with (results / "m1_emission_check.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    for label, option in options.items():
        print(label, json.dumps({key: value for key, value in option.items() if key != "cases"}))


if __name__ == "__main__":
    main()
