import argparse
import json
from pathlib import Path

from kerr_sbi.config import load_config
from kerr_sbi.evaluation import evaluate_run, load_protocol, summarize
from kerr_sbi.persistence import exclusive_lock, write_json

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "evaluation.toml"


def valid_name(value: str) -> str:
    if not value or not all(c.isascii() and (c.isalnum() or c in "-_") for c in value):
        raise argparse.ArgumentTypeError("use only ASCII letters, digits, hyphens or underscores")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate completed physical runs on common development observations and the "
            "fixed numerical reference cases; the first run is the paired baseline"
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--name", type=valid_name, required=True, help="output: runs/NAME")
    parser.add_argument("--runs", type=valid_name, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("runs must be distinct")
    root = load_config()["project_root"]
    protocol = load_protocol(args.protocol)
    directory = root / "runs" / args.name
    if directory.exists() and not (directory / "evaluation.json").exists():
        parser.error(f"runs/{args.name} exists and is not an evaluation; choose a fresh name")
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".lock"):
        record = {"protocol": protocol, "runs": args.runs}
        frozen = directory / "evaluation.json"
        if frozen.exists() and json.loads(frozen.read_text()) != record:
            raise ValueError("this evaluation name already uses a different protocol or run list")
        write_json(frozen, record)
        for run in args.runs:
            completion = evaluate_run(root, run, protocol, directory / run)
            print(f"Evaluated {run} in {completion['elapsed_seconds']:.1f} s", flush=True)
        output = summarize(directory, args.runs, protocol)
    primary = str(protocol["sampling"]["posterior_seeds"][0])
    print(
        json.dumps(
            {
                run: {
                    key: output["models"][run]["repetitions"][primary][key]
                    for key in ("nll", "prior_nll", "rmse")
                }
                for run in args.runs
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
