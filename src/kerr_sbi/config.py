import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

Config = dict[str, Any]
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    cfg["project_root"] = path.parent.parent
    return cfg


def project_path(cfg: Config, name: str) -> Path:
    return Path(cfg["project_root"]) / cfg["paths"][name]


def dataset_config(cfg: Config, name: str | None) -> Config:
    result = deepcopy(cfg)
    if name is not None:
        if not name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in name):
            raise ValueError(
                "dataset name must contain only ASCII letters, digits, hyphens or underscores"
            )
        result["paths"]["data"] = str(Path(cfg["paths"]["data"]) / "datasets" / name)
        result["paths"]["figures"] = str(Path(result["paths"]["data"]) / "figures")
    return result


def isolate_run(cfg: Config, name: str, *, plot_only: bool = False) -> Config:
    if not name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in name):
        raise ValueError("run name must contain only ASCII letters, digits, hyphens or underscores")
    isolated = deepcopy(cfg)
    for key in ("data", "results", "figures"):
        isolated["paths"][key] = str(Path(cfg["paths"][key]) / name)
    if not plot_only:
        for key in ("data", "results", "figures"):
            path = project_path(isolated, key)
            if path.exists():
                raise FileExistsError(f"run output already exists: {path}")
    return isolated
