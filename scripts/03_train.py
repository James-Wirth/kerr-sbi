import argparse
import json
from pathlib import Path

from kerr_sbi.config import DEFAULT_CONFIG, dataset_config, load_config
from kerr_sbi.experiments import experiment_config, preflight
from kerr_sbi.inference import posterior_check
from kerr_sbi.telemetry import JsonlEvents
from kerr_sbi.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--profile", choices=("default", "local-smoke", "local-pilot"), default="default"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--sample-check", action="store_true")
    args = parser.parse_args()
    if args.profile != "default" and (args.dummy or args.overfit):
        parser.error(
            "named profiles are for noisy physical training; overfit and dummy modes are separate"
        )
    if args.dataset and args.dummy:
        parser.error("--dataset selects physical data; dummy data has a separate location")
    if not args.preflight and not args.run:
        parser.error("--run is required unless --preflight is selected")
    if args.sample_check and (not args.dummy or args.overfit):
        parser.error("--sample-check requires noisy --dummy training")
    cfg = experiment_config(dataset_config(load_config(args.config), args.dataset), args.profile)
    if args.preflight:
        print(json.dumps(preflight(cfg, dummy=args.dummy, overfit=args.overfit), indent=2))
        return
    result = run_training(
        cfg,
        args.run,
        dummy=args.dummy,
        overfit=args.overfit,
        resume=args.resume,
        stop_after=args.stop_after,
        on_event=JsonlEvents(),
    )
    print(json.dumps(result, indent=2))
    if args.sample_check and result["stop_reason"] in ("complete", "early_stopping"):
        print(json.dumps(posterior_check(cfg, args.run), indent=2))
    if result["stop_reason"] == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
