import math
from pathlib import Path
from typing import BinaryIO

import numpy as np


def _read_header_line(stream: BinaryIO) -> bytes:
    while line := stream.readline():
        line = line.strip()
        if line and not line.startswith(b"#"):
            return line
    raise ValueError("incomplete PFM header")


def read_pfm(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as stream:
        magic = _read_header_line(stream)
        if magic not in (b"PF", b"Pf"):
            raise ValueError(f"unsupported PFM format: {magic!r}")
        try:
            width, height = map(int, _read_header_line(stream).split())
            scale = float(_read_header_line(stream))
        except (ValueError, OverflowError) as exc:
            raise ValueError("invalid PFM dimensions or scale") from exc
        if width <= 0 or height <= 0:
            raise ValueError("PFM dimensions must be positive")
        if not math.isfinite(scale) or scale == 0:
            raise ValueError("PFM scale must be finite and nonzero")
        payload = stream.read()
    channels = 3 if magic == b"PF" else 1
    expected_bytes = width * height * channels * np.dtype("f4").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"PFM payload has {len(payload)} bytes; expected {expected_bytes}")
    dtype = np.dtype("<f4" if scale < 0 else ">f4")
    shape = (height, width, channels) if channels == 3 else (height, width)
    image = np.frombuffer(payload, dtype=dtype).reshape(shape)
    image = np.array(image[::-1], dtype=np.float32, order="C", copy=True)
    image *= abs(scale)
    return image
