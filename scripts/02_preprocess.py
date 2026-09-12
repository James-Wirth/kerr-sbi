import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from kerr_sbi.config import DEFAULT_CONFIG, Config, dataset_config, load_config, project_path
from kerr_sbi.dataset import load_preprocessed, preprocess_dataset


def save_montage(cfg: Config, split: str) -> Path:
    arrays, metadata = load_preprocessed(cfg, split)
    settings = cfg["dataset"]
    count = min(len(arrays["idx"]), settings["montage_max_images"])
    columns = settings["montage_columns"]
    if count < 1 or columns < 1:
        raise ValueError("montage requires available images and a positive column count")
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.3 * columns, 2.4 * rows), layout="constrained", squeeze=False
    )
    for axis in axes.flat:
        axis.set_axis_off()
    for row, axis in enumerate(axes.flat[:count]):
        x = arrays["x"][row]
        a, inclination = arrays["theta"][row]
        axis.imshow(
            np.arcsinh(x / (settings["montage_peak_fraction"] * x.max())),
            cmap="magma",
            origin="upper",
        )
        axis.set_title(f"#{arrays['idx'][row]:04d} · a={a:.3f} · i={inclination:.1f}°", fontsize=9)
    sampling = metadata["identity"]["scene"]["supersample"]
    fig.suptitle(
        f"M3 {split} smoke test · {count} images · {sampling}×{sampling} subpixel rays\n"
        "PSF and normalized flux; no noise. Independent asinh display scales."
    )
    directory = project_path(cfg, "figures") / "m3"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{split}_smoke.{suffix}", dpi=settings["figure_dpi"])
    plt.close(fig)
    return directory / f"{split}_smoke.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--montage", action="store_true")
    args = parser.parse_args()
    cfg = dataset_config(load_config(args.config), args.dataset)
    metadata = preprocess_dataset(cfg, args.split)
    summary = {
        key: metadata[key]
        for key in (
            "split",
            "n_requested",
            "n_rendered",
            "n_failed",
            "failed_indices",
            "missing_indices",
            "invalid_indices",
        )
    }
    if args.montage:
        summary["montage"] = str(save_montage(cfg, args.split))
    print(json.dumps(summary, indent=2))
    if metadata["n_failed"] or metadata["invalid_indices"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
