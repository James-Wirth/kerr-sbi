import tomllib
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
