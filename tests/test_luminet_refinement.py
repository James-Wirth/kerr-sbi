import importlib
import json
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kerr_sbi.transfer import geometry_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("refine_luminet")


def test_continuation_changes_only_declared_trace_controls(driver):
    original = importlib.import_module("replicate_luminet")
    settings = tomllib.loads((ROOT / "configs/luminet.toml").read_text())
    parent = dict(settings=settings, renders=original.plan(settings))
    before = deepcopy(parent)
    rows = driver.render_plan(parent)
    assert parent == before
    assert len(rows) == 3
    base = next(r["scene"] for r in parent["renders"] if r["name"] == "R4000_n7")
    assert [r["scene"]["camera"]["jitter"] for r in rows] == [False, True, False]
    assert [r["scene"]["integrator"]["tol"] for r in rows] == [1e-9, 1e-9, 1e-11]
    for row in rows:
        scene = deepcopy(row["scene"])
        assert scene["camera"]["supersample"] == scene["camera"]["supersample_max"] == 9
        for key in ["supersample", "supersample_max", "jitter"]:
            scene["camera"][key] = base["camera"][key]
        scene["integrator"]["tol"] = base["integrator"]["tol"]
        assert scene == base


def test_missing_cache_cannot_spend_render_budget(driver, tmp_path):
    cache = SimpleNamespace(root=tmp_path, identity={"binary_sha256": "test"})
    scene = {"metric": {"spin": 0}, "disk": {"r_in": 0, "r_out": 40}}
    with pytest.raises(FileNotFoundError, match="rerender forbidden"):
        driver.require_cached(cache, scene)
    directory = tmp_path / geometry_key(scene, cache.identity)
    directory.mkdir()
    (directory / "cache.json").write_text(json.dumps({}))
    driver.require_cached(cache, scene)


def test_invalid_node_fails_even_with_zero_residual(driver):
    a = dict(b=np.ones((3, 181)), z=np.zeros((3, 181)), valid=np.ones((3, 181), bool))
    b = deepcopy(a)
    b["valid"][0, 0] = False
    gates = dict(b_rms=0.01, b_max=0.03, z_rms=0.001, z_max=0.003)
    result = driver.compare(a, b, gates)
    assert not result["passed"]
    assert not result["rings"][0]["passed"]
    assert all(row["passed"] for row in result["rings"][1:])
