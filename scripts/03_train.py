import argparse
import json
from pathlib import Path

from kerr_sbi.config import DEFAULT_CONFIG, load_config
from kerr_sbi.inference import posterior_check
from kerr_sbi.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--sample-check", action="store_true")
    args = parser.parse_args()
    if args.sample_check and (not args.dummy or args.overfit):
        parser.error("--sample-check requires noisy --dummy training")
    cfg = load_config(args.config)
    result = run_training(
        cfg,
        args.run,
        dummy=args.dummy,
        overfit=args.overfit,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    print(json.dumps(result, indent=2))
    if args.sample_check and result["stop_reason"] in ("complete", "early_stopping"):
        print(json.dumps(posterior_check(cfg, args.run), indent=2))
    if result["stop_reason"] == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
