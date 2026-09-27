import json
from pathlib import Path

import pytest

from kerr_sbi.evidence import check_claims, resolve_pointer
from kerr_sbi.persistence import file_sha256


def test_pointer_resolves_nested_objects_lists_and_escapes() -> None:
    document = {"models": {"a/b": [{"nll": 1.5}]}, "~key": 2}
    assert resolve_pointer(document, "/models/a~1b/0/nll") == 1.5
    assert resolve_pointer(document, "/~0key") == 2
    assert resolve_pointer(document, "") is document
    with pytest.raises(ValueError):
        resolve_pointer(document, "models")


def test_claim_checks_report_passes_failures_and_errors(tmp_path: Path) -> None:
    evidence = tmp_path / "runs/study/summary.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps({"nll": 0.898375, "flags": [[True, False], [True]], "cases": [1, 2, 3]})
    )
    file = "runs/study/summary.json"
    claims = [
        {
            "id": "nll",
            "file": file,
            "pointer": "/nll",
            "check": "approx",
            "expected": 0.89838,
            "tolerance": 1e-5,
        },
        {
            "id": "nll-tight",
            "file": file,
            "pointer": "/nll",
            "check": "approx",
            "expected": 0.89838,
            "tolerance": 1e-7,
        },
        {"id": "accepted", "file": file, "pointer": "/flags", "check": "count_true", "expected": 2},
        {"id": "cases", "file": file, "pointer": "/cases", "check": "length", "expected": 3},
        {"id": "hash", "file": file, "check": "sha256", "expected": file_sha256(evidence)},
        {"id": "no-test", "check": "absent", "paths": ["data/datasets/*/test"]},
        {
            "id": "missing",
            "file": "runs/other.json",
            "pointer": "/x",
            "check": "equals",
            "expected": 1,
        },
    ]
    results = {result["id"]: result for result in check_claims(tmp_path, claims)}
    assert [name for name, result in results.items() if result["passed"]] == [
        "nll",
        "accepted",
        "cases",
        "hash",
        "no-test",
    ]
    assert "error" in results["missing"]
    (tmp_path / "data/datasets/pilot/test").mkdir(parents=True)
    assert not check_claims(tmp_path, [claims[5]])[0]["passed"]
    with pytest.raises(ValueError, match="unique"):
        check_claims(tmp_path, [claims[0], claims[0]])
