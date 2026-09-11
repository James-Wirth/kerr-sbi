import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import tomli_w

from kerr_sbi.config import Config, project_path


class RenderError(RuntimeError):
    def __init__(self, message: str, stderr: str = "") -> None:
        self.stderr = stderr
        super().__init__(f"{message}\n{stderr}".rstrip())


def nullgeo_binary() -> str:
    return os.environ.get("NULLGEO_BIN", "nullgeo")


def template_sha256(cfg: Config) -> str:
    return hashlib.sha256(project_path(cfg, "scene_template").read_bytes()).hexdigest()


def build_scene(a: float, incl_deg: float, output_path: Path, cfg: Config) -> str:
    prior, scene, emission = cfg["prior"], cfg["scene"], cfg["emission"]
    if not math.isfinite(a) or not prior["a_min"] <= a <= prior["a_max"]:
        raise ValueError("spin lies outside the configured prior")
    if not math.isfinite(incl_deg) or not (
        prior["incl_min_deg"] <= incl_deg <= prior["incl_max_deg"]
    ):
        raise ValueError("inclination lies outside the configured prior")
    if scene["width"] != 64 or scene["height"] != 64:
        raise ValueError("v1 requires 64 by 64 images")
    if emission["model"] == "stylized":
        disk_fields = {key: emission[key] for key in ("g_power", "emissivity_index")}
    elif emission["model"] == "blackbody":
        disk_fields = {"t_in": emission["t_in"]}
    else:
        raise ValueError(f"unsupported disk model: {emission['model']}")
    angle = math.radians(incl_deg)
    quoted_path = tomli_w.dumps({"path": str(output_path.resolve())}).split(" = ", 1)[1].strip()
    result = (
        project_path(cfg, "scene_template")
        .read_text()
        .format(
            **scene,
            a=float(a),
            camera_x=-scene["camera_distance"] * math.sin(angle),
            camera_z=scene["camera_distance"] * math.cos(angle),
            disk_model=emission["model"],
            disk_fields=tomli_w.dumps(disk_fields).strip(),
            output_path=quoted_path,
        )
    )
    if cfg["simulator"]["smoke_png"]:
        result += "\n" + tomli_w.dumps(
            {"output": [{"path": str(output_path.with_suffix(".png").resolve()), "format": "png"}]}
        )
    return result


def render(a: float, incl_deg: float, out_stem: str | Path, cfg: Config) -> Path:
    destination = Path(f"{out_stem}.pfm").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary = shutil.which(nullgeo_binary())
    if binary is None:
        raise RenderError(f"nullgeo executable not found: {nullgeo_binary()}")
    with tempfile.TemporaryDirectory(prefix=".nullgeo-", dir=destination.parent) as directory:
        temporary = Path(directory)
        pfm_path = temporary / "image.pfm"
        scene_path = temporary / "scene.toml"
        scene_path.write_text(build_scene(a, incl_deg, pfm_path, cfg))
        try:
            completed = subprocess.run(
                [binary, "render", str(scene_path)],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=cfg["simulator"]["timeout_seconds"],
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            raise RenderError(f"nullgeo timed out for a={a}, i={incl_deg}", stderr) from exc
        except OSError as exc:
            raise RenderError(f"could not execute nullgeo: {exc}") from exc
        if completed.returncode:
            raise RenderError(
                f"nullgeo exited {completed.returncode} for a={a}, i={incl_deg}",
                completed.stderr,
            )
        if not pfm_path.is_file() or pfm_path.stat().st_size == 0:
            raise RenderError("nullgeo succeeded without a nonempty PFM", completed.stderr)
        if cfg["simulator"]["smoke_png"]:
            png_path = pfm_path.with_suffix(".png")
            if not png_path.is_file():
                raise RenderError("nullgeo did not write the requested smoke PNG", completed.stderr)
            png_path.replace(destination.with_suffix(".png"))
        pfm_path.replace(destination)
    return destination
