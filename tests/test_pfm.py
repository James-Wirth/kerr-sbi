from pathlib import Path

import numpy as np
import pytest

from kerr_sbi.pfm import read_pfm


@pytest.mark.parametrize("magic, shape", [(b"PF", (2, 3, 3)), (b"Pf", (2, 3))])
@pytest.mark.parametrize("scale", [-1.0, 1.0, -2.5, 2.5])
def test_endianness_row_order_and_scale(
    tmp_path: Path, magic: bytes, shape: tuple[int, ...], scale: float
) -> None:
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 0.25
    dtype = "<f4" if scale < 0 else ">f4"
    path = tmp_path / "image.pfm"
    path.write_bytes(
        magic + b"\n3 2\n" + f"{scale}\n".encode() + expected[::-1].astype(dtype).tobytes()
    )
    actual = read_pfm(path)
    np.testing.assert_array_equal(actual, expected * abs(scale))
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


@pytest.mark.parametrize(
    "content",
    [
        b"P6\n1 1\n-1\n",
        b"PF\n",
        b"PF\n0 2\n-1\n",
        b"PF\n1 1\n0\n",
        b"PF\n1 1\nnan\n",
        b"PF\n1 1\n-1\n" + b"\0" * 11,
        b"PF\n1 1\n-1\n" + b"\0" * 13,
    ],
)
def test_rejects_invalid_pfm(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.pfm"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_pfm(path)
