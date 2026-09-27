import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kerr_sbi.persistence import file_sha256

CHECKS = ("equals", "approx", "length", "count_true", "absent", "sha256")


def resolve_pointer(document: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def approximately(observed: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(approximately(o, e, tolerance) for o, e in zip(observed, expected, strict=True))
        )
    return isinstance(observed, int | float) and math.isclose(
        observed, expected, rel_tol=0, abs_tol=tolerance
    )


def count_true(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, list):
        return sum(count_true(item) for item in value)
    raise ValueError("count_true expects booleans or nested lists of booleans")


def check_claim(root: Path, claim: Mapping[str, Any]) -> dict[str, Any]:
    kind = claim["check"]
    if kind not in CHECKS:
        raise ValueError(f"unknown check {kind!r}")
    if kind == "absent":
        present = sorted(
            path.relative_to(root).as_posix()
            for pattern in claim["paths"]
            for path in root.glob(pattern)
        )
        return {"id": claim["id"], "passed": not present, "observed": present}
    path = root / claim["file"]
    if kind == "sha256":
        observed = file_sha256(path)
        return {"id": claim["id"], "passed": observed == claim["expected"], "observed": observed}
    observed = resolve_pointer(json.loads(path.read_text()), claim.get("pointer", ""))
    if kind == "length":
        observed = len(observed)
    elif kind == "count_true":
        observed = count_true(observed)
    if kind == "approx":
        passed = approximately(observed, claim["expected"], claim["tolerance"])
    else:
        passed = observed == claim["expected"]
    return {"id": claim["id"], "passed": passed, "observed": observed}


def check_claims(root: Path, claims: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identifiers = [claim["id"] for claim in claims]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("claim identifiers must be unique")
    results = []
    for claim in claims:
        try:
            results.append(check_claim(root, claim))
        except (OSError, KeyError, IndexError, ValueError, TypeError) as exc:
            results.append({"id": claim["id"], "passed": False, "error": repr(exc)})
    return results
