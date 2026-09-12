import json
import time
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

EventSink = Callable[[dict[str, Any]], None]


class TrainingEvents:
    def __init__(self, sink: EventSink | None, started: float):
        self.sink = sink
        self.started = started
        self.previous_seconds = 0.0
        self.step = 0
        self.active = False
        self.attempt = uuid.uuid4().hex

    def __enter__(self) -> "TrainingEvents":
        return self

    def emit(self, kind: str, **values: Any) -> None:
        self.step = values.get("step", self.step)
        self.active = self.active or kind == "started"
        if self.sink is None:
            return
        record = {
            "schema_version": 1,
            "attempt": self.attempt,
            "timestamp": time.time(),
            "elapsed_seconds": self.previous_seconds + time.perf_counter() - self.started,
            "kind": kind,
            "status": "running",
            "step": self.step,
            **values,
        }
        try:
            self.sink(record)
        except Exception as exc:
            warnings.warn(f"Training telemetry disabled: {exc}", RuntimeWarning, stacklevel=2)
            self.sink = None

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is not None and self.active:
            self.emit("failed", status="failed", error=f"{type(error).__name__}: {error}")


class JsonlEvents:
    def __init__(self, interval_seconds: float = 1.0):
        if interval_seconds < 0:
            raise ValueError("event interval must be nonnegative")
        self.interval_seconds = interval_seconds
        self.last_update = -float("inf")
        self.path: Path | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        if event["kind"] == "started":
            self.path = Path(event["directory"]) / "events.jsonl"
            self.last_update = -float("inf")
        if self.path is None:
            return
        now = time.monotonic()
        if event["kind"] == "update":
            if now - self.last_update < self.interval_seconds:
                return
            self.last_update = now
        with self.path.open("a+b") as stream:
            if event["kind"] == "started" and stream.tell():
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write((json.dumps(event, allow_nan=False) + "\n").encode())
