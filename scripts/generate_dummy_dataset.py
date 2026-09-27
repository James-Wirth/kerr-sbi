import argparse
import json
from pathlib import Path

from kerr_sbi.config import DEFAULT_CONFIG, load_config
from kerr_sbi.dummy import build_dummy_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the tiny nonphysical dataset used for software checks"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_dummy_dataset(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
