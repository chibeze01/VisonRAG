from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricPoint:
    count: int
    total: float


class InMemoryMetrics:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._hist: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._hist[name].append(value)

    def snapshot(self) -> dict[str, MetricPoint]:
        output: dict[str, MetricPoint] = {}
        with self._lock:
            for name, values in self._hist.items():
                output[name] = MetricPoint(count=len(values), total=float(sum(values)))
            for name, value in self._counters.items():
                output[f"{name}.count"] = MetricPoint(count=value, total=float(value))
        return output

