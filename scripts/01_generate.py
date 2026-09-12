import argparse
import json
from pathlib import Path

from kerr_sbi.config import DEFAULT_CONFIG, load_config
from kerr_sbi.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        summary = generate_dataset(load_config(args.config), args.split, args.n, args.seed)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume from validated outputs.")
        raise SystemExit(130) from None
    print(json.dumps(summary, indent=2))
    if summary["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
