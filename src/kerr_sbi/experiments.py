from copy import deepcopy
from dataclasses import asdict
from typing import Any

from kerr_sbi.config import Config
from kerr_sbi.data import fit_normalization, load_split, training_rows
from kerr_sbi.train import effective_config, optimizer_for, overfit_settings


def experiment_config(cfg: Config, profile: str) -> Config:
    result = deepcopy(cfg)
    if profile == "default":
        return result
    if profile not in ("local-smoke", "local-pilot"):
        raise ValueError("unknown training profile")
    result["training"].update(result[profile.replace("-", "_")])
    return result


def preflight(cfg: Config, *, dummy: bool, overfit: bool) -> dict[str, Any]:
    effective = effective_config(cfg, dummy=dummy, overfit=overfit)
    data = load_split(effective, "train", allow_dummy=dummy)
    train, validation = training_rows(data, effective["training"]["validation_fraction"])
    if overfit:
        count = overfit_settings(effective, dummy)["subset_size"]
        if not 2 <= count <= len(train):
            raise ValueError("overfit subset must fit inside training-only rows")
        train = train[:count]
    stats = fit_normalization(data, train, effective)
    training = effective["training"]
    batch = training["batch_size"]
    if type(batch) is not int or batch < 1:
        raise ValueError("invalid batch size")
    batches = (len(train) + batch - 1) // batch
    steps = batches * training["max_epochs"]
    if training["max_steps"]:
        steps = min(steps, training["max_steps"])
    optimizer_for(effective, steps)
    return {
        "dummy": dummy,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "batch_size": batch,
        "batches_per_epoch": batches,
        "total_steps": steps,
        "warmup_steps": training["warmup_steps"],
        "sigma_n": overfit_settings(effective, dummy)["sigma_n"]
        if overfit
        else effective["observation"]["sigma_n"],
        "normalization": asdict(stats),
        "scope": "software smoke check" if len(train) < 256 else "training experiment",
    }
