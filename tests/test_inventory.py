import os
from pathlib import Path

from kerr_sbi.inventory import catalog_coverage, compare, relocate, snapshot, unit_of


def make_tree(root: Path) -> None:
    for relative, text in {
        "data/datasets/pilot/train/x.npy": "images",
        "data/m1/a.pfm": "render",
        "runs/study/summary.json": "{}",
        "runs/study/__pycache__/cached.pyc": "ignored",
        "logs/run.log": "log",
        "runs/.study.lock": "",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_units_group_datasets_runs_and_logs() -> None:
    assert unit_of("data/datasets/pilot/train/x.npy") == "data/datasets/pilot"
    assert unit_of("data/m1/a.pfm") == "data/m1"
    assert unit_of("runs/study/deep/file.npz") == "runs/study"
    assert unit_of("logs/run.log") == "logs"
    assert unit_of("results/m0_checks.json") == "results/m0_checks.json"


def test_snapshot_skips_bytecode_and_counts_hardlinks_once(tmp_path: Path) -> None:
    make_tree(tmp_path)
    before = snapshot(tmp_path, ("data", "runs", "logs"))
    assert "runs/study/__pycache__/cached.pyc" not in before["files"]
    os.link(tmp_path / "data/m1/a.pfm", tmp_path / "data/m1/b.pfm")
    after = snapshot(tmp_path, ("data", "runs", "logs"))
    assert after["logical_bytes"] == before["logical_bytes"] + len("render")
    assert after["disk_bytes"] == before["disk_bytes"]


def test_compare_accepts_declared_moves_and_deletions_only(tmp_path: Path) -> None:
    make_tree(tmp_path)
    before = snapshot(tmp_path, ("data", "runs", "logs"))
    (tmp_path / "archive").mkdir()
    (tmp_path / "runs/study").rename(tmp_path / "archive/study")
    (tmp_path / "logs/run.log").unlink()
    after = snapshot(tmp_path, ("data", "runs", "logs", "archive"))
    undeclared = compare(before, after)
    assert undeclared["missing"] == ["logs/run.log", "runs/study/summary.json"]
    declared = compare(before, after, {"runs/study": "archive/study"}, ["logs/run.log"])
    assert declared["missing"] == [] and declared["changed"] == []
    assert declared["moved"] == 1 and declared["deleted_as_declared"] == 1
    (tmp_path / "data/m1/a.pfm").write_text("edited")
    edited = compare(
        before,
        snapshot(tmp_path, ("data", "runs", "logs", "archive")),
        {"runs/study": "archive/study"},
        ["logs/run.log"],
    )
    assert edited["changed"] == ["data/m1/a.pfm"]


def test_relocate_uses_the_longest_matching_prefix() -> None:
    moves = {"runs/a": "archive/a", "runs/a/sub": "elsewhere/sub"}
    assert relocate("runs/a/file", moves) == "archive/a/file"
    assert relocate("runs/a/sub/file", moves) == "elsewhere/sub/file"
    assert relocate("runs/ab/file", moves) == "runs/ab/file"


def test_catalog_reports_uncatalogued_and_empty_entries(tmp_path: Path) -> None:
    make_tree(tmp_path)
    units = snapshot(tmp_path, ("data", "runs", "logs"))["units"]
    catalog = {
        "entries": [
            {"path": "data", "class": "active"},
            {"path": "runs/study", "class": "historical"},
            {"path": "runs/gone", "class": "disposable"},
        ]
    }
    result = catalog_coverage(catalog, units)
    assert result["uncatalogued_units"] == ["logs"]
    assert result["empty_entries"] == ["runs/gone"]
    assert result["hidden_units"] == ["runs/.study.lock"]
    entry = next(item for item in result["entries"] if item["path"] == "runs/study")
    assert entry["files"] == 1 and entry["content_sha256"] == units["runs/study"]["content_sha256"]
