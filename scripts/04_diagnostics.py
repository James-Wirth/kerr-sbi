import argparse
import json
from pathlib import Path

from kerr_sbi.config import DEFAULT_CONFIG, load_config
from kerr_sbi.diagnostics import development_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory physical development checks using saved best weights"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    report = development_check(load_config(args.config), args.run)
    print(
        json.dumps({key: value for key, value in report.items() if key != "provenance"}, indent=2)
    )


if __name__ == "__main__":
    main()
