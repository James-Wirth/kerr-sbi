import argparse
import json
from pathlib import Path

from kerr_sbi.evidence import check_claims

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check documented claims against the saved experiment evidence"
    )
    parser.add_argument("claims", type=Path, help="JSON file with a top-level 'claims' list")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    claims = json.loads(args.claims.read_text())["claims"]
    results = check_claims(args.root.resolve(), claims)
    statements = {claim["id"]: claim["statement"] for claim in claims}
    for result in results:
        mark = "ok  " if result["passed"] else "FAIL"
        detail = result.get("error", result.get("observed"))
        print(f"{mark} {result['id']}: {statements[result['id']]} [{json.dumps(detail)[:120]}]")
    failed = sum(not result["passed"] for result in results)
    print(f"{len(results) - failed}/{len(results)} claims match their evidence")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
