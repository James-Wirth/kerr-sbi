from copy import deepcopy
from pathlib import Path

import pytest

from kerr_sbi.config import Config, isolate_run, project_path


def test_run_isolation_preserves_outputs_and_refuses_reuse(cfg: Config, tmp_path: Path) -> None:
    cfg["project_root"] = tmp_path
    original = deepcopy(cfg)
    isolated = isolate_run(cfg, "confirmation")
    for key in ("data", "results", "figures"):
        assert project_path(isolated, key) == project_path(cfg, key) / "confirmation"
    output = project_path(isolated, "results")
    output.mkdir(parents=True)
    archive = output / "saved.npz"
    archive.write_bytes(b"preserved")
    with pytest.raises(FileExistsError, match="already exists"):
        isolate_run(cfg, "confirmation")
    assert isolate_run(cfg, "confirmation", plot_only=True) == isolated
    assert archive.read_bytes() == b"preserved"
    assert cfg == original


@pytest.mark.parametrize("name", ["", ".", "..", "../old", "/old", "a/b", "a b"])
def test_run_name_cannot_escape_output_location(cfg: Config, name: str) -> None:
    with pytest.raises(ValueError, match="run name"):
        isolate_run(cfg, name)
