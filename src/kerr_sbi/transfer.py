import hashlib
import json
import subprocess
import time
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tomli_w

from kerr_sbi.visibility import UAS_TO_RAD, image_frequencies, translation_phase

SAMPLE_DTYPE = np.dtype(
    [(name, "<u8") for name in ("pixel_index", "sample_index")]
    + [(name, "<f8") for name in ("offset_x", "offset_y", "screen_u", "screen_v", "weight")]
    + [("render_weight", "<f4"), ("radius", "<f8"), ("g", "<f8"), ("radiance", "<f4")]
    + [
        (name, "u1")
        for name in (
            "has_intersection",
            "radius_valid",
            "g_available",
            "g_valid",
            "radiance_valid",
            "outcome",
            "finished",
        )
    ]
    + [("steps_accepted", "<u8"), ("steps_rejected", "<u8")]
)
BUILD_FIELDS = (
    "package_version",
    "build_id",
    "git_revision",
    "git_state_at_build",
    "source_digest",
    "rustc",
    "target",
    "profile",
    "parallel",
)
TRACE_FIELDS = (
    "rtol",
    "atol",
    "dl_init",
    "dl_min",
    "dl_max",
    "escape_radius",
    "max_steps",
    "camera_energy",
    "camera_time",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def geometry_scene(scene: dict) -> dict:
    result = deepcopy(scene)
    result.pop("output", None)
    for name in ("emissivity_index", "g_power"):
        result["disk"].pop(name, None)
    return result


def geometry_key(scene: dict, binary_identity: dict) -> str:
    return canonical_hash(
        {"schema_version": 1, "binary": binary_identity, "scene": geometry_scene(scene)}
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    indices = indices.copy()
    value = np.zeros(len(indices))
    denominator = 1.0
    while np.any(indices):
        denominator *= base
        value += (indices % base) / denominator
        indices //= base
    return value


@dataclass(frozen=True)
class Transfer:
    samples: np.ndarray
    metadata: dict
    scene: dict

    @classmethod
    def load(cls, directory: Path) -> "Transfer":
        directory = Path(directory)
        meta = tomllib.loads((directory / "metadata.toml").read_text())
        require(
            meta.get("schema") == "nullgeo.thin-disk-transfer" and meta.get("schema_version") == 1,
            "unsupported transfer schema/version",
        )
        require(
            meta.get("array") == "samples.npy" and meta.get("source_scene") == "scene.toml",
            "unexpected transfer filenames",
        )
        samples = np.load(directory / "samples.npy", allow_pickle=False, mmap_mode="r")
        require(
            samples.dtype == SAMPLE_DTYPE and meta["record_bytes"] == 103,
            "unsupported transfer record layout",
        )
        require(samples.shape == (meta["sample_count"],), "sample count mismatch")
        scene = tomllib.loads((directory / "scene.toml").read_text())
        result = cls(samples, meta, scene)
        result.validate()
        return result

    def validate(self) -> None:
        s, m = self.samples, self.metadata
        scene = m["resolved_scene"]
        camera, disk = scene["camera"], scene["disk"]
        require(
            scene["metric"]["kind"] in ("kerr", "schwarzschild") and disk["model"] == "stylized",
            "unsupported transfer physics",
        )
        for name in ("sky", "sky_secondary"):
            if name in scene:
                require(scene[name].get("uniform") == [0.0, 0.0, 0.0], "nonblack sky")
        require(m["sampling"] == "camera-grid-v1-adaptive-replacement", "unknown sampling")
        for name in BUILD_FIELDS + TRACE_FIELDS:
            require(name in m, f"missing metadata {name}")
        w, h = m["width"], m["height"]
        require(
            w > 0 and h > 0 and camera["width"] == w and camera["height"] == h, "invalid dimensions"
        )
        require(0 < m["effective_r_in"] < m["effective_r_out"], "invalid annulus")
        require(np.all(s["pixel_index"] < w * h), "pixel index out of bounds")
        ids = s["pixel_index"].astype(np.intp)
        counts = np.bincount(ids, minlength=w * h)
        n, nmax = camera["supersample"], camera["supersample_max"]
        require(1 <= n <= nmax, "invalid sampling controls")
        allowed = [n * n, nmax * nmax]
        require(np.isin(counts, allowed).all(), "invalid contributing counts")
        require(np.array_equal(ids, np.repeat(np.arange(w * h), counts)), "pixel ordering")
        starts = np.cumsum(counts) - counts
        order = np.arange(len(s)) - np.repeat(starts, counts)
        require(np.array_equal(s["sample_index"], order), "replacement sample ordering")
        sizes = np.sqrt(counts[ids]).astype(int)
        dx = radical_inverse(order + 1, 2) if camera["jitter"] else 0.5
        dy = radical_inverse(order + 1, 3) if camera["jitter"] else 0.5
        for field, expected in (
            ("offset_x", (order % sizes + dx) / sizes),
            ("offset_y", (order // sizes + dy) / sizes),
            ("weight", 1.0 / counts[ids]),
            ("render_weight", np.float32(1) / counts[ids].astype(np.float32)),
        ):
            require(np.allclose(s[field], expected, rtol=0, atol=2e-16), f"invalid {field}")
        scale = np.tan(np.deg2rad(camera["fov_deg"]) / 2)
        expected_u = (2 * ((ids % w + s["offset_x"]) / w) - 1) * scale
        expected_v = (1 - 2 * ((ids // w + s["offset_y"]) / h)) * scale * h / w
        require(
            np.allclose(s["screen_u"], expected_u, rtol=1e-14, atol=1e-16)
            and np.allclose(s["screen_v"], expected_v, rtol=1e-14, atol=1e-16),
            "screen orientation/coordinates mismatch",
        )
        for name in (
            "has_intersection",
            "radius_valid",
            "g_available",
            "g_valid",
            "radiance_valid",
            "finished",
        ):
            require(np.isin(s[name], [0, 1]).all(), f"invalid mask {name}")
        hit, available = s["has_intersection"].astype(bool), s["g_available"].astype(bool)
        require(np.isin(s["outcome"], np.arange(6)).all(), "unknown outcome")
        require(np.array_equal(s["finished"], s["outcome"] < 4), "finished/outcome mismatch")
        require(np.array_equal(s["outcome"] == 3, hit), "opaque outcome/intersection mismatch")
        require(np.all(~available | hit), "g without intersection")
        require(
            np.isnan(s["radius"][~hit]).all() and np.isnan(s["g"][~available]).all(),
            "missing value sentinel mismatch",
        )
        require(
            np.array_equal(s["radius_valid"], hit & np.isfinite(s["radius"]) & (s["radius"] > 0)),
            "radius validity mismatch",
        )
        require(
            np.array_equal(s["g_valid"], available & np.isfinite(s["g"]) & (s["g"] > 0)),
            "g validity mismatch",
        )
        require(
            np.array_equal(s["radiance_valid"], np.isfinite(s["radiance"])),
            "radiance validity mismatch",
        )
        valid = s["radius_valid"].astype(bool)
        require(
            np.all(
                (s["radius"][valid] >= m["effective_r_in"])
                & (s["radius"][valid] <= m["effective_r_out"])
            ),
            "hit outside annulus",
        )

    def diagnostics(self) -> dict:
        s = self.samples
        counts = np.bincount(s["pixel_index"].astype(np.intp))
        unique, frequency = np.unique(counts, return_counts=True)
        return {
            "samples": len(s),
            "pixels_by_sample_count": dict(zip(map(str, unique), map(int, frequency), strict=True)),
            "outcomes": {str(k): int(np.sum(s["outcome"] == k)) for k in range(6)},
            "unfinished": int(np.sum(s["finished"] == 0)),
            "invalid_radius": int(np.sum(s["has_intersection"] > s["radius_valid"])),
            "invalid_g": int(np.sum(s["g_available"] > s["g_valid"])),
            "missing_g_at_hit": int(np.sum(s["has_intersection"] > s["g_available"])),
            "invalid_radiance": int(np.sum(s["radiance_valid"] == 0)),
        }

    def intensity(
        self, emissivity_index: float | None = None, g_power: float | None = None
    ) -> np.ndarray:
        s, disk = self.samples, self.metadata["resolved_scene"]["disk"]
        q = disk["emissivity_index"] if emissivity_index is None else emissivity_index
        power = disk["g_power"] if g_power is None else g_power
        require(np.isfinite([q, power]).all(), "nonfinite emission parameters")
        require(
            np.array_equal(s["radius_valid"], s["has_intersection"])
            and np.array_equal(s["g_valid"], s["g_available"])
            and np.all(s["radiance_valid"]),
            "invalid emission quantities; inspect diagnostics",
        )
        valid = s["radius_valid"].astype(bool) & s["g_valid"].astype(bool)
        values = np.zeros(len(s), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            values[valid] = (
                s["g"][valid] ** power
                * (s["radius"][valid] / self.metadata["effective_r_in"]) ** -q
            )
        require(np.isfinite(values).all(), "emission overflow")
        return values

    def pixels(self, values: np.ndarray | None = None) -> np.ndarray:
        values = self.samples["radiance"] if values is None else np.asarray(values)
        require(values.shape == (len(self.samples),), "one intensity per sample required")
        result = np.zeros(self.metadata["width"] * self.metadata["height"], dtype=np.float32)
        weighted = values.astype(np.float32) * self.samples["render_weight"]
        np.add.at(result, self.samples["pixel_index"].astype(np.intp), weighted)
        return result.reshape(self.metadata["height"], self.metadata["width"])

    def visibility(
        self,
        uv: np.ndarray,
        fov_uas: float,
        emissivity_index: float | None = None,
        g_power: float | None = None,
        flux_jy: float | None = 1.0,
        angle_deg: float = 0.0,
        offset_uas: tuple[float, float] = (0.0, 0.0),
        baseline_batch: int = 64,
        sample_batch: int = 16384,
    ) -> np.ndarray:
        require(np.all(self.samples["finished"]), "unfinished rays; visibility is not certified")
        require(
            isinstance(baseline_batch, int)
            and baseline_batch > 0
            and isinstance(sample_batch, int)
            and sample_batch > 0,
            "invalid batch sizes",
        )
        frequencies = image_frequencies(uv, fov_uas / self.metadata["width"], angle_deg)
        phase = translation_phase(np.asarray(uv), offset_uas)
        scale = 2 * np.tan(np.deg2rad(self.metadata["resolved_scene"]["camera"]["fov_deg"]) / 2)
        xy = np.column_stack((self.samples["screen_u"], -self.samples["screen_v"]))
        xy *= self.metadata["width"] / scale
        masses = self.intensity(emissivity_index, g_power) * self.samples["weight"]
        masses *= (fov_uas * UAS_TO_RAD / self.metadata["width"]) ** 2
        if flux_jy is not None:
            require(
                np.isfinite(flux_jy) and flux_jy > 0 and masses.sum() > 0,
                "positive finite flux and nonzero emission required",
            )
            masses *= flux_jy / masses.sum()
        result = np.zeros(len(frequencies), dtype=np.complex128)
        for start in range(0, len(masses), sample_batch):
            stop = start + sample_batch
            mass = masses[start:stop]
            if not np.any(mass):
                continue
            nonzero = mass != 0
            nodes, mass = xy[start:stop][nonzero], mass[nonzero]
            for b in range(0, len(result), baseline_batch):
                f = frequencies[b : b + baseline_batch]
                argument = f[:, :1] * nodes[:, 0] + f[:, 1:] * nodes[:, 1]
                result[b : b + baseline_batch] += np.exp(-2j * np.pi * argument) @ mass
        return result * phase


class GeometryCache:
    def __init__(self, root: Path, binary: Path, binary_identity: dict) -> None:
        self.root, self.binary = Path(root), Path(binary).resolve()
        self.identity = deepcopy(binary_identity)
        require(sha256(self.binary) == self.identity["binary_sha256"], "binary identity mismatch")
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, scene: dict, timeout: float = 180.0) -> Transfer:
        require(sha256(self.binary) == self.identity["binary_sha256"], "binary changed")
        key = geometry_key(scene, self.identity)
        directory = self.root / key
        manifest_path = directory / "cache.json"
        if directory.exists():
            record = json.loads(manifest_path.read_text())
            require(
                record["key"] == key and record["binary"] == self.identity,
                "cache identity mismatch",
            )
            for name, digest in record["files"].items():
                require(sha256(directory / name) == digest, f"cache corruption: {name}")
            transfer = Transfer.load(directory / "transfer")
            require(geometry_scene(transfer.scene) == geometry_scene(scene), "cache scene mismatch")
            require(
                record["resolved_key"] == self.resolved_key(transfer), "resolved cache mismatch"
            )
            return transfer
        directory.mkdir()
        rendered = deepcopy(scene)
        rendered["output"] = [{"path": str(directory.resolve() / "image.pfm"), "format": "pfm"}]
        config = directory / "scene.toml"
        config.write_text(tomli_w.dumps(rendered))
        command = [
            str(self.binary),
            "render",
            str(config.resolve()),
            "--transfer-export",
            str(directory.resolve() / "transfer"),
        ]
        start = time.perf_counter()
        with (directory / "render.log").open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout
            )
        seconds = time.perf_counter() - start
        transfer = Transfer.load(directory / "transfer")
        if "source_commit" in self.identity:
            require(
                transfer.metadata["git_revision"] == self.identity["source_commit"],
                "export source commit mismatch",
            )
        record = {
            "key": key,
            "binary": self.identity,
            "command": command,
            "seconds": seconds,
            "resolved_key": self.resolved_key(transfer),
            "build": {k: transfer.metadata[k] for k in BUILD_FIELDS},
            "files": {
                str(p.relative_to(directory)): sha256(p)
                for p in directory.rglob("*")
                if p.is_file()
            },
        }
        manifest_path.write_text(json.dumps(record, indent=2) + "\n")
        return transfer

    def resolved_key(self, transfer: Transfer) -> str:
        meta = transfer.metadata
        return canonical_hash(
            {
                "binary": self.identity,
                "schema_version": 1,
                "scene": geometry_scene(meta["resolved_scene"]),
                "controls": {k: meta[k] for k in TRACE_FIELDS},
                "effective_r_in": meta["effective_r_in"],
                "effective_r_out": meta["effective_r_out"],
                "build": {k: meta[k] for k in BUILD_FIELDS},
            }
        )
