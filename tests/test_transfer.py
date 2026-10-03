import json
from copy import deepcopy

import numpy as np
import pytest
import tomli_w

from kerr_sbi.transfer import (
    BUILD_FIELDS,
    SAMPLE_DTYPE,
    TRACE_FIELDS,
    GeometryCache,
    Transfer,
    geometry_key,
    sha256,
)
from kerr_sbi.visibility import UAS_TO_RAD, direct_visibility


def fixture_transfer(tmp_path, n=1, nmax=1, jitter=False):
    w, h = 3, 2
    camera = dict(
        width=w,
        height=h,
        supersample=n,
        supersample_max=nmax,
        jitter=jitter,
        fov_deg=22.0,
        position=[-60.0, 0.0, 60.0],
    )
    scene = dict(
        camera=camera,
        metric=dict(kind="kerr", mass=1.0, spin=0.7),
        disk=dict(model="stylized", r_in=8.0, r_out=15.0, emissivity_index=2.0, g_power=3.0),
        sky=dict(uniform=[0.0, 0.0, 0.0]),
    )
    counts = np.array([n * n, nmax * nmax] * 3)
    samples = np.zeros(sum(counts), dtype=SAMPLE_DTYPE)
    pos = 0
    for pixel, count in enumerate(counts):
        size = int(np.sqrt(count))
        for k in range(count):
            dx, dy = 0.5, 0.5
            if jitter:
                fractions = []
                for base in (2, 3):
                    value, index, denominator = 0.0, k + 1, 1
                    while index:
                        denominator *= base
                        value += (index % base) / denominator
                        index //= base
                    fractions.append(value)
                dx, dy = fractions
            x, y = (k % size + dx) / size, (k // size + dy) / size
            scale = np.tan(np.deg2rad(camera["fov_deg"]) / 2)
            samples[pos] = (
                pixel,
                k,
                x,
                y,
                (2 * ((pixel % w + x) / w) - 1) * scale,
                (1 - 2 * ((pixel // w + y) / h)) * scale * h / w,
                1 / count,
                np.float32(1) / np.float32(count),
                8.0,
                1.0,
                1.0,
                1,
                1,
                1,
                1,
                1,
                3,
                1,
                30,
                2,
            )
            pos += 1
    meta = dict(
        schema="nullgeo.thin-disk-transfer",
        schema_version=1,
        width=w,
        height=h,
        sample_count=len(samples),
        record_bytes=103,
        array="samples.npy",
        source_scene="scene.toml",
        sampling="camera-grid-v1-adaptive-replacement",
        effective_r_in=8.0,
        effective_r_out=15.0,
        resolved_scene=scene,
        **dict.fromkeys(BUILD_FIELDS, "test"),
        **dict.fromkeys(TRACE_FIELDS, 1.0),
    )
    np.save(tmp_path / "samples.npy", samples)
    (tmp_path / "metadata.toml").write_text(tomli_w.dumps(meta))
    (tmp_path / "scene.toml").write_text(tomli_w.dumps(scene))
    return Transfer.load(tmp_path)


@pytest.mark.parametrize(
    "n,nmax,jitter",
    [(1, 1, False), (1, 1, True), (3, 3, False), (3, 3, True), (1, 3, False), (2, 3, True)],
)
def test_layout_and_float32_reconstruction(tmp_path, n, nmax, jitter):
    transfer = fixture_transfer(tmp_path, n, nmax, jitter)
    values = np.linspace(0.1, 3.1, len(transfer.samples))
    expected = np.zeros(6, dtype=np.float32)
    for s, value in zip(transfer.samples, values, strict=True):
        expected[s["pixel_index"]] += np.float32(value) * s["render_weight"]
    np.testing.assert_array_equal(transfer.pixels(values).ravel(), expected)
    assert transfer.diagnostics()["samples"] == len(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_index", 10),
        ("pixel_index", 99),
        ("weight", 0.5),
        ("screen_v", 99),
        ("finished", 0),
        ("g_valid", 0),
        ("outcome", 9),
    ],
)
def test_corruption_rejected(tmp_path, field, value):
    transfer = fixture_transfer(tmp_path)
    samples = transfer.samples.copy()
    samples[field][0] = value
    np.save(tmp_path / "samples.npy", samples)
    with pytest.raises(ValueError):
        Transfer.load(tmp_path)


def test_version_and_dtype_rejected(tmp_path):
    transfer = fixture_transfer(tmp_path)
    meta = deepcopy(transfer.metadata)
    meta["schema_version"] = 2
    (tmp_path / "metadata.toml").write_text(tomli_w.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        Transfer.load(tmp_path)
    meta["schema_version"] = 1
    (tmp_path / "metadata.toml").write_text(tomli_w.dumps(meta))
    np.save(tmp_path / "samples.npy", np.zeros(len(transfer.samples)))
    with pytest.raises(ValueError, match="layout"):
        Transfer.load(tmp_path)


def test_failures_and_invalid_values_preserved(tmp_path):
    transfer = fixture_transfer(tmp_path)
    s = transfer.samples.copy()
    s["outcome"] = np.arange(6)
    s["finished"] = s["outcome"] < 4
    hit = s["outcome"] == 3
    for field in ("has_intersection", "radius_valid", "g_available", "g_valid"):
        s[field] = hit
    s["radius"][~hit] = np.nan
    s["g"][~hit] = np.nan
    s["radiance"][~hit] = 0
    np.save(tmp_path / "samples.npy", s)
    loaded = Transfer.load(tmp_path)
    assert len(loaded.samples) == 6
    assert loaded.diagnostics()["unfinished"] == 2
    np.testing.assert_array_equal(loaded.samples["weight"], s["weight"])
    np.testing.assert_array_equal(loaded.pixels().ravel(), s["radiance"])
    with pytest.raises(ValueError, match="unfinished"):
        loaded.visibility(np.zeros((1, 2)), 125)
    s["g"][hit] = np.inf
    s["g_valid"][hit] = 0
    np.save(tmp_path / "samples.npy", s)
    loaded = Transfer.load(tmp_path)
    assert np.isinf(loaded.samples["g"][hit]).all()
    with pytest.raises(ValueError, match="invalid emission"):
        loaded.intensity()


def test_missing_g_is_black_and_no_hit_weight_renormalization(tmp_path):
    transfer = fixture_transfer(tmp_path)
    s = transfer.samples.copy()
    s["g_available"][0] = s["g_valid"][0] = 0
    s["g"][0] = np.nan
    s["radiance"][0] = 0
    np.save(tmp_path / "samples.npy", s)
    transfer = Transfer.load(tmp_path)
    assert transfer.intensity().sum() == 5
    raw = transfer.visibility(np.zeros((1, 2)), 6, flux_jy=None)
    np.testing.assert_allclose(raw, 5 * (2 * UAS_TO_RAD) ** 2, rtol=1e-14)


def test_visibility_orientation_rotation_flux_batches_and_no_sinc(tmp_path):
    transfer = fixture_transfer(tmp_path)
    s = transfer.samples.copy()
    s["g"][:] = 0.01
    s["g"][2] = 1.0
    changed = Transfer(s, transfer.metadata, transfer.scene)
    uv = np.array([[0, 0], [2e9, 3e9], [-2e9, -3e9]])
    for angle in (0, 90, 31):
        for offset in ((0, 0), (3, -5)):
            actual = changed.visibility(
                uv,
                6,
                flux_jy=0.6,
                angle_deg=angle,
                offset_uas=offset,
                baseline_batch=1,
                sample_batch=2,
            )
            other = changed.visibility(uv, 6, flux_jy=0.6, angle_deg=angle, offset_uas=offset)
            np.testing.assert_allclose(actual, other, atol=2e-15)
            image = changed.intensity().reshape(2, 3)
            theta = np.deg2rad(angle)
            rotated_uv = uv @ [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            aperture = np.sinc(rotated_uv * 2 * UAS_TO_RAD).prod(axis=1)
            expected = 0.6 * direct_visibility(image, uv, 2, angle, offset) / aperture
            np.testing.assert_allclose(actual, expected, atol=2e-15)
            assert abs(actual[0] - 0.6) < 2e-15
            assert abs(actual[1] - actual[2].conjugate()) < 2e-15
    point = changed.visibility(uv, 6, g_power=100, angle_deg=90, flux_jy=0.6)
    expected = 0.6 * np.exp(-2j * np.pi * (uv @ [-1, 2]) * UAS_TO_RAD)
    np.testing.assert_allclose(point, expected, atol=2e-15)


def test_cache_identity_covers_geometry_and_omits_only_emission_output(tmp_path):
    transfer = fixture_transfer(tmp_path)
    scene = transfer.scene
    identity = dict(binary_sha256="abc", source_commit="123")
    key = geometry_key(scene, identity)
    changed = deepcopy(scene)
    changed["disk"].update(emissivity_index=3.0, g_power=4.0)
    changed["output"] = [{"path": "elsewhere"}]
    assert geometry_key(changed, identity) == key
    for section, field, value in [
        ("metric", "spin", 0.0),
        ("disk", "r_in", 8.1),
        ("disk", "r_out", 16.0),
        ("camera", "supersample", 3),
        ("camera", "jitter", True),
        ("camera", "position", [0, 0, 85]),
        ("integrator", "tol", 1e-10),
    ]:
        changed = deepcopy(scene)
        changed.setdefault(section, {})[field] = value
        assert geometry_key(changed, identity) != key
    assert geometry_key(scene, dict(binary_sha256="different")) != key


def test_cache_hit_and_corruption_without_retrace(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    transfer = fixture_transfer(fixture_dir)
    binary = tmp_path / "binary"
    binary.write_bytes(b"fake")
    identity = dict(binary_sha256=sha256(binary))
    cache = GeometryCache(tmp_path / "cache", binary, identity)
    key = geometry_key(transfer.scene, identity)
    target = cache.root / key
    target.mkdir()
    fixture_dir.rename(target / "transfer")
    files = {str(p.relative_to(target)): sha256(p) for p in target.rglob("*") if p.is_file()}
    (target / "cache.json").write_text(
        json.dumps(
            dict(key=key, binary=identity, files=files, resolved_key=cache.resolved_key(transfer))
        )
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("unexpected retrace"))
    changed = deepcopy(transfer.scene)
    changed["disk"]["emissivity_index"] = 3.1
    assert cache.get(changed).intensity(3.1).shape == (6,)
    with (target / "transfer/samples.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corruption"):
        cache.get(changed)
