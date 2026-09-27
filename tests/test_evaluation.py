from pathlib import Path

import numpy as np
import pytest

from kerr_sbi.evaluation import (
    bin_rows,
    check_prior_support,
    interval_statistics,
    load_protocol,
    paired_bootstrap,
    wilson,
)

PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "evaluation.toml"
LEVELS = (0.68, 0.9, 0.95)


def test_wilson_known_midpoint_and_endpoints() -> None:
    np.testing.assert_allclose(wilson(5, 10), [0.2365930905, 0.7634069095], atol=1e-9)
    np.testing.assert_allclose(wilson(0, 10)[0], 0, atol=1e-15)
    np.testing.assert_allclose(wilson(10, 10)[1], 1, atol=1e-15)
    assert wilson([0, 5, 10], 10).shape == (3, 2)
    with pytest.raises(ValueError):
        wilson(0, 0)
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_interval_statistics_known_uniform_draws() -> None:
    truth = np.array([[0.5, 50.0], [0.5, 50.0]])
    theta = np.stack([np.linspace(0, 1, 1001), np.linspace(40, 60, 1001)], axis=-1)[:, None, :]
    theta = np.repeat(theta, 2, axis=1)
    report, per = interval_statistics(
        theta, truth, np.array([1.0, 3.0]), np.array([4.0, 4.0]), LEVELS
    )
    np.testing.assert_allclose(report["rmse"], [0, 0], atol=1e-12)
    np.testing.assert_allclose(report["mean_width90"], [0.9, 18.0])
    assert report["nll"] == 2 and report["nll_gain"] == 2
    np.testing.assert_array_equal(per["coverage_0.9"], np.ones((2, 2)))
    assert report["coverage"]["0.68"]["count"] == [2, 2]


def test_paired_bootstrap_preserves_pairing_and_rmse_order() -> None:
    baseline = {
        "nll": np.arange(10.0),
        "squared_error": np.tile([1.0, 4.0], (10, 1)),
        "width90": np.ones((10, 2)),
    }
    current = {
        "nll": baseline["nll"] + 2,
        "squared_error": np.tile([4.0, 9.0], (10, 1)),
        "width90": np.ones((10, 2)) * 3,
    }
    for level in LEVELS:
        baseline[f"coverage_{level}"] = np.ones((10, 2))
        current[f"coverage_{level}"] = np.zeros((10, 2))
    indices = np.random.default_rng(1).integers(0, 10, (100, 10))
    result = paired_bootstrap(current, baseline, indices, LEVELS)
    assert result["nll"]["difference"] == 2
    np.testing.assert_array_equal(result["nll"]["paired_bootstrap95"], [2, 2])
    np.testing.assert_array_equal(result["rmse"]["difference"], [1, 1])
    np.testing.assert_array_equal(result["coverage_0.9"]["difference"], [-1, -1])
    same = paired_bootstrap(baseline, baseline, indices, LEVELS)
    for item in same.values():
        np.testing.assert_array_equal(item["difference"], np.zeros_like(item["difference"]))
        np.testing.assert_array_equal(
            item["paired_bootstrap95"], np.zeros_like(item["paired_bootstrap95"])
        )


def test_bins_include_only_the_final_upper_edge() -> None:
    truth = np.array([[0.0, 5.0], [0.1, 10.0], [0.5, 30.0], [0.98, 80.0], [0.98, 80.0]])
    theta = np.repeat(truth[None], 3, axis=0)
    analysis = {
        "coverage_levels": list(LEVELS),
        "headline_coverage_level": 0.9,
        "inclination_bin_edges_deg": [5, 30, 80],
        "spin_bin_edges": [0, 0.5, 0.98],
    }
    rows = bin_rows("run", theta, truth, np.ones(5), np.full(5, 2.0), analysis)
    counts = {(row["bin_variable"], row["lower"]): row["n"] for row in rows}
    assert counts == {
        ("inclination", 5): 2,
        ("inclination", 30): 3,
        ("spin", 0): 2,
        ("spin", 0.5): 3,
    }
    with pytest.raises(ValueError, match="fewer than two"):
        bin_rows(
            "run",
            theta,
            truth,
            np.ones(5),
            np.full(5, 2.0),
            analysis | {"inclination_bin_edges_deg": [5, 6, 80]},
        )


def test_protocol_refuses_the_reserved_test_split(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["development"]["split"] == "dev"
    text = PROTOCOL.read_text().replace('split = "dev"', 'split = "test"')
    (tmp_path / "test.toml").write_text(text)
    with pytest.raises(ValueError, match="reserved test split"):
        load_protocol(tmp_path / "test.toml")
    repeated = PROTOCOL.read_text().replace(
        "posterior_seeds = [20260926, 2026092701]", "posterior_seeds = [1, 1]"
    )
    (tmp_path / "repeated.toml").write_text(repeated)
    with pytest.raises(ValueError, match="unique"):
        load_protocol(tmp_path / "repeated.toml")


def test_prior_support_rejects_draws_outside_the_physical_prior() -> None:
    prior = {"a_min": 0.0, "a_max": 0.98, "incl_min_deg": 5.0, "incl_max_deg": 80.0}
    check_prior_support(np.array([[[0.0, 5.0], [0.98, 80.0]]]), prior)
    for bad in ([0.99, 40.0], [0.5, 4.0], [np.nan, 40.0]):
        with pytest.raises(ValueError):
            check_prior_support(np.array([[bad]]), prior)
