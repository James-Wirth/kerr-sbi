from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kerr_sbi.fisher import central_difference, fisher_bounds
from kerr_sbi.persistence import file_sha256, write_json
from kerr_sbi.reference import RenderCache


def disk_edge_check(
    directory: Path, cfg: dict, settings: dict, *, cache_directory: Path | None = None
) -> dict:
    cache_directory = directory if cache_directory is None else cache_directory
    options = settings["fixed_radius"]
    if options["r_in"] < 6 * cfg["scene"]["mass"]:
        raise ValueError("fixed radius must lie outside every prograde ISCO in the prior")
    destination = directory / "disk_edge"
    destination.mkdir(exist_ok=True)
    changed = deepcopy(cfg)
    changed["scene"]["r_in"] = options["r_in"]
    caches = [
        RenderCache(cache_directory / "renders", cfg),
        RenderCache(cache_directory / "fixed_radius_renders", changed),
    ]
    curves, bounds, derivatives = [], [], []
    sigma = cfg["observation"]["sigma_n"]
    for label, cache in zip(("ISCO", "fixed"), caches, strict=True):
        images = np.empty(
            (len(options["inclinations_deg"]), len(options["spins"]), 64, 64), dtype=np.float32
        )
        for i, inclination in enumerate(options["inclinations_deg"]):
            for j, a in enumerate(options["spins"]):
                images[i, j] = cache.image(a, inclination)
            print(f"Disk edge {label}: spin curve at i={inclination}", flush=True)
        curves.append(images)
        jacobians = []
        for factor in (1.0, 0.5):
            local = []
            for a, inclination in options["derivative_points"]:
                da, di = factor * options["delta_a"], factor * options["delta_incl_deg"]
                local.append(
                    np.stack(
                        [
                            central_difference(
                                cache.image(a - da, inclination),
                                cache.image(a + da, inclination),
                                da,
                            ).ravel(),
                            central_difference(
                                cache.image(a, inclination - di),
                                cache.image(a, inclination + di),
                                di,
                            ).ravel(),
                        ],
                        axis=-1,
                    )
                )
            jacobians.append(np.stack(local))
            print(f"Disk edge {label}: derivative step factor {factor}", flush=True)
        derivatives.append(np.stack(jacobians))
        bounds.append(fisher_bounds(np.stack(jacobians), sigma))
    curves, derivatives = np.stack(curves), np.stack(derivatives)
    separation = (
        np.linalg.norm(
            (curves - curves[:, :, :1]).reshape(
                2, len(options["inclinations_deg"]), len(options["spins"]), -1
            ),
            axis=-1,
        )
        / sigma
    )
    relative_derivative_change = np.linalg.norm(
        derivatives[:, 1] - derivatives[:, 0], axis=2
    ) / np.linalg.norm(derivatives[:, 0], axis=2)
    information_ratio = np.sum(derivatives[1] ** 2, axis=2) / np.sum(derivatives[0] ** 2, axis=2)
    bound_ratio = bounds[1]["marginal_std"] / bounds[0]["marginal_std"]
    report = {
        "settings": options,
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("disk_edge.py", "reference.py", "fisher.py", "obs_model.py")
        },
        "sigma_n": sigma,
        "interpretation": (
            "Paired forward-model comparison; fixed inner radius is known, not marginalized. "
            "Separation is full-image Mahalanobis distance at known inclination; half its square "
            "is Gaussian expected log likelihood ratio. Local information ratios change multiple "
            "physical image features and are not additive fractions of ISCO information."
        ),
        "spin_separation_from_zero": separation.tolist(),
        "conditional_information_ratio_fixed_over_isco": information_ratio.tolist(),
        "joint_std_ratio_fixed_over_isco": bound_ratio.tolist(),
        "derivative_relative_change": relative_derivative_change.tolist(),
        "baseline_marginal_std": bounds[0]["marginal_std"].tolist(),
        "fixed_marginal_std": bounds[1]["marginal_std"].tolist(),
        "numerically_stable": bool(np.all(relative_derivative_change < 0.2)),
    }
    np.savez(
        destination / "comparison.npz",
        images=curves,
        derivatives=derivatives,
        separation=separation,
    )
    write_json(destination / "summary.json", report)
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for j, inclination in enumerate(options["inclinations_deg"]):
        line = panels[0].plot(options["spins"], separation[0, j], label=f"{inclination:g}°")[0]
        panels[0].plot(options["spins"], separation[1, j], "--", color=line.get_color())
    panels[0].set(
        xlabel="Spin a (compared with a=0)",
        ylabel="Image separation / noise σ",
        title="Solid: disk at ISCO · dashed: inner radius 8M",
    )
    panels[0].legend(title="Inclination", fontsize=8)
    x = np.arange(len(options["derivative_points"]))
    panels[1].bar(x - 0.18, information_ratio[0, :, 0], 0.36, label="Original step")
    panels[1].bar(x + 0.18, information_ratio[1, :, 0], 0.36, label="Half step")
    panels[1].set(
        xticks=x,
        xticklabels=[f"a={a:.2f}\ni={i:g}°" for a, i in options["derivative_points"]],
        ylabel="Conditional spin Fisher information ratio",
        title=(
            "Fixed inner radius / disk at ISCO"
            if report["numerically_stable"]
            else "Exploratory ratios: spin derivatives\nhave not converged"
        ),
        yscale="log",
    )
    panels[1].legend(fontsize=8)
    fig.savefig(destination / "disk_edge_comparison.png", dpi=settings["reference"]["figure_dpi"])
    plt.close(fig)
    return report
