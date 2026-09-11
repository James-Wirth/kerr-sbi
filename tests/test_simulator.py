import json
import math
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest

from kerr_sbi.config import Config
from kerr_sbi.obs_model import gaussian_blur, luminance, preprocess
from kerr_sbi.pfm import read_pfm
from kerr_sbi.simulator import RenderError, build_scene, nullgeo_binary, render


@pytest.mark.parametrize("incl_deg", [5.0, 45.0, 80.0])
@pytest.mark.parametrize("model", ["stylized", "blackbody"])
def test_template_camera_and_emission_fields(
    tmp_path: Path, cfg: Config, incl_deg: float, model: str
) -> None:
    cfg["emission"]["model"] = model
    output = tmp_path / 'output with "quotes".pfm'
    scene = tomllib.loads(build_scene(0.9, incl_deg, output, cfg))
    angle = math.radians(incl_deg)
    np.testing.assert_allclose(
        scene["camera"]["position"], [-85 * math.sin(angle), 0.0, 85 * math.cos(angle)]
    )
    assert scene["metric"] == {"kind": "kerr", "mass": 1.0, "spin": 0.9}
    assert scene["camera"]["supersample"] == scene["camera"]["supersample_max"] == 3
    assert scene["sky"] == {"uniform": [0.0, 0.0, 0.0]}
    assert scene["output"] == [{"path": str(output), "format": "pfm"}]
    assert "integrator" not in scene
    model_keys = {"g_power", "emissivity_index"} if model == "stylized" else {"t_in"}
    assert scene["disk"].keys() == {"model", "r_in", "r_out"} | model_keys


def test_optional_smoke_png(tmp_path: Path, cfg: Config) -> None:
    cfg["simulator"]["smoke_png"] = True
    scene = tomllib.loads(build_scene(0.9, 80.0, tmp_path / "image.pfm", cfg))
    assert scene["output"][1] == {"path": str(tmp_path / "image.png"), "format": "png"}


@pytest.mark.parametrize("a, incl_deg", [(-0.1, 45), (1.0, 45), (0.5, 0), (0.5, 90)])
def test_prior_limits(tmp_path: Path, cfg: Config, a: float, incl_deg: float) -> None:
    with pytest.raises(ValueError, match="prior"):
        build_scene(a, incl_deg, tmp_path / "image.pfm", cfg)


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_output", "interrupt"])
def test_failed_render_preserves_existing_output_and_cleans_temporary_files(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    destination = tmp_path / "image.pfm"
    destination.write_bytes(b"previous output")
    monkeypatch.setattr("kerr_sbi.simulator.shutil.which", lambda _: "/fake/nullgeo")

    def failed_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        scene = tomllib.loads(Path(command[2]).read_text())
        assert command[:2] == ["/fake/nullgeo", "render"]
        assert kwargs["timeout"] == 120.0
        assert kwargs["capture_output"] is True
        assert kwargs["cwd"] == Path(command[2]).parent
        temporary_output = Path(scene["output"][0]["path"])
        assert temporary_output != destination
        if failure != "missing_output":
            temporary_output.write_bytes(b"partial output")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 120, stderr=b"timeout stderr")
        if failure == "interrupt":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0, "", "stderr")

    monkeypatch.setattr("kerr_sbi.simulator.subprocess.run", failed_run)
    error_type = KeyboardInterrupt if failure == "interrupt" else RenderError
    with pytest.raises(error_type) as caught:
        render(0.9, 80.0, tmp_path / "image", cfg)
    if failure in ("exit", "timeout"):
        assert "stderr" in caught.value.stderr
    assert destination.read_bytes() == b"previous output"
    assert sorted(tmp_path.iterdir()) == [destination]


@pytest.mark.slow
def test_real_render_orientation_determinism_and_preprocessing(tmp_path: Path, cfg: Config) -> None:
    if shutil.which(nullgeo_binary()) is None:
        pytest.skip("requires nullgeo on PATH or NULLGEO_BIN")
    cfg["simulator"]["smoke_png"] = True
    elapsed = []
    images = []
    for name in ("first", "repeat"):
        started = time.perf_counter()
        path = render(0.9, 80.0, tmp_path / name, cfg)
        elapsed.append(time.perf_counter() - started)
        images.append(read_pfm(path))
        assert path.with_suffix(".png").is_file()
    rgb, repeat = images
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.float32
    assert np.all(np.isfinite(rgb))
    np.testing.assert_array_equal(rgb, repeat)
    assert np.all(rgb[[0, 0, -1, -1], [0, -1, 0, -1]] == 0)
    y = luminance(rgb)
    left_flux = y[:, :32].sum(dtype=np.float64)
    right_flux = y[:, 32:].sum(dtype=np.float64)
    assert left_flux > 10 * right_flux
    obs = cfg["observation"]
    blurred = gaussian_blur(y, obs["sigma_psf"], obs["kernel_size"])
    blur_relative_flux_change = abs(blurred.sum(dtype=np.float64) / y.sum(dtype=np.float64) - 1)
    assert blur_relative_flux_change < 1e-4
    x = preprocess(rgb, cfg)
    assert x.sum(dtype=np.float64) == pytest.approx(4096, rel=1e-7)
    print(
        json.dumps(
            {
                "left_right_flux_ratio": float(left_flux / right_flux),
                "blur_relative_flux_change": float(blur_relative_flux_change),
                "processed_flux": float(x.sum(dtype=np.float64)),
                "peak_luminance": float(y.max()),
                "render_seconds": elapsed,
                "image_path": str(tmp_path / "first.pfm"),
                "bit_identical": True,
            },
            indent=2,
        )
    )
