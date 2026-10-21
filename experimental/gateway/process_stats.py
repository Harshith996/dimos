# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dtop's numbers for the blueprint view: the run's /resource_stats (published under `dtop`), heard only while asked for."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

TOPIC = "/resource_stats"
HISTORY = 60  # cpu samples per process (the monitor sends one a second)
IDLE_S = 30.0  # nobody asked this long: unsubscribe
STALE_S = 3.0


class ProcessStats:
    def __init__(self, transport_factory: Any = None) -> None:
        self._factory = transport_factory
        self._lock = threading.Lock()
        self._transport: Any = None
        self._latest: dict[str, Any] | None = None
        self._heard = 0.0
        self._asked = 0.0
        self._cpu: dict[str, deque[float]] = {}

    def _on_msg(self, msg: dict[str, Any]) -> None:
        with self._lock:
            self._latest = msg
            self._heard = time.monotonic()
            for key, cpu in cpu_samples(msg).items():
                self._cpu.setdefault(key, deque(maxlen=HISTORY)).append(cpu)

    def _ensure_subscribed(self) -> None:
        if self._transport is not None:
            return
        if self._factory is None:
            from dimos.core.transport_factory import make_transport

            self._factory = make_transport
        transport = self._factory(TOPIC)
        transport.subscribe(self._on_msg)
        transport.start()
        self._transport = transport

    def stop_if_idle(self) -> None:
        with self._lock:
            transport = self._transport
            if transport is None or time.monotonic() - self._asked < IDLE_S:
                return
            self._transport = None
            self._latest = None
            self._cpu.clear()
        transport.stop()

    def snapshot(self) -> dict[str, Any]:
        """The newest stats with each process's cpu history; `up` false until the first message."""
        with self._lock:
            self._asked = time.monotonic()
            self._ensure_subscribed()
            if self._latest is None:
                return {
                    "up": False,
                    "age": None,
                    "stale": False,
                    "coordinator": None,
                    "workers": [],
                    "cpuHistory": {},
                }
            age = time.monotonic() - self._heard
            return {
                "up": True,
                "age": round(age, 2),
                "stale": age > STALE_S,
                "coordinator": self._latest.get("coordinator"),
                "workers": self._latest.get("workers") or [],
                "cpuHistory": {key: list(values) for key, values in self._cpu.items()},
            }


def cpu_samples(msg: dict[str, Any]) -> dict[str, float]:
    """cpu% per process: `coordinator`, `worker:<id>`, and `child:<pid>` for a worker's children."""
    out: dict[str, float] = {}
    coordinator = msg.get("coordinator") or {}
    if coordinator:
        out["coordinator"] = float(coordinator.get("cpu_percent") or 0.0)
    for worker in msg.get("workers") or []:
        out[f"worker:{worker.get('worker_id')}"] = float(worker.get("cpu_percent") or 0.0)
        for child in worker.get("children") or []:
            out[f"child:{child.get('pid')}"] = float(child.get("cpu_percent") or 0.0)
    return out
