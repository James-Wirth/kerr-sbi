import argparse
import json
from pathlib import Path

from kerr_sbi.config import DEFAULT_CONFIG, Config, dataset_config, load_config, project_path
from kerr_sbi.dataset import generate_dataset, load_preprocessed, preprocess_dataset
from kerr_sbi.diagnostics import development_check
from kerr_sbi.experiments import experiment_config, preflight
from kerr_sbi.persistence import exclusive_lock
from kerr_sbi.telemetry import JsonlEvents
from kerr_sbi.train import run_training


def prepare_split(cfg: Config, split: str, count: int, seed: int) -> None:
    summary = generate_dataset(cfg, split, count, seed)
    if summary["n_failed"]:
        raise RuntimeError(f"{split} has failed renders; resume generation before training")
    directory = project_path(cfg, "data") / split
    if (directory / "meta.json").exists():
        _, metadata = load_preprocessed(cfg, split)
        if metadata["n_requested"] == count and metadata["n_rendered"] == count:
            return
    metadata = preprocess_dataset(cfg, split)
    if metadata["n_rendered"] != count:
        raise RuntimeError(f"{split} is incomplete after preprocessing")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reusable local images, overfit check, noisy pilot and development diagnostics"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = dataset_config(load_config(args.config), args.dataset)
    if not args.run or not all(c.isascii() and (c.isalnum() or c in "-_") for c in args.run):
        parser.error("run name must contain only ASCII letters, digits, hyphens or underscores")
    names = (f"{args.run}_overfit", args.run)
    if not args.resume and any((project_path(cfg, "runs") / name).exists() for name in names):
        parser.error("run outputs already exist; use --resume or a new run name")
    runs = project_path(cfg, "runs")
    with (
        exclusive_lock(runs / f".{args.run}.job.lock"),
        exclusive_lock(runs / f".{names[0]}.job.lock"),
    ):
        run_pilot(cfg, args.run)


def run_pilot(cfg: Config, run_name: str) -> None:
    names = (f"{run_name}_overfit", run_name)
    settings = cfg["local_pilot_data"]
    for split in ("train", "dev"):
        prepare_split(cfg, split, settings[f"{split}_count"], settings[f"{split}_seed"])
    print(json.dumps(preflight(cfg, dummy=False, overfit=True), indent=2), flush=True)
    overfit = run_training(
        cfg,
        names[0],
        overfit=True,
        resume=(project_path(cfg, "runs") / names[0]).exists(),
        on_event=JsonlEvents(),
    )
    if overfit["stop_reason"] != "overfit_target":
        raise RuntimeError(
            f"overfit target was not met: {overfit['stop_reason']}; inspect {names[0]}"
        )
    pilot = experiment_config(cfg, "local-pilot")
    print(json.dumps(preflight(pilot, dummy=False, overfit=False), indent=2), flush=True)
    interval = settings["checkpoint_steps"]
    if type(interval) is not int or interval < 1:
        raise ValueError("checkpoint_steps must be positive")
    while True:
        directory = project_path(pilot, "runs") / run_name
        step = 0
        if (directory / "latest.json").exists():
            pointer = json.loads((directory / "latest.json").read_text())
            metadata = json.loads((directory / pointer["directory"] / "metadata.json").read_text())
            step = metadata["summary"]["step"]
        result = run_training(
            pilot,
            run_name,
            resume=directory.exists(),
            stop_after=step + interval,
            on_event=JsonlEvents(),
        )
        if result["stop_reason"] != "paused":
            break
    if result["stop_reason"] not in ("complete", "early_stopping"):
        raise RuntimeError(f"pilot stopped before completion: {result['stop_reason']}")
    report = development_check(cfg, run_name)
    print(
        json.dumps({key: value for key, value in report.items() if key != "provenance"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Interrupted. Rerun with --resume to reuse validated images and published checkpoints."
        )
        raise SystemExit(130) from None
