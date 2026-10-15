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

"""The native binary end to end: health records, the run's logs, robot_health passed through, and a bad_event dump."""

from collections.abc import Callable, Iterator
import json
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from dimos.core.transport import LCMTransport
from dimos.diagnostics.system_log.module import SystemLog
from dimos.msgs.std_msgs.Bool import Bool
from dimos.msgs.std_msgs.String import String

_TIMEOUT_S = 30.0


class _Records:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._changed = threading.Condition()

    def add(self, msg: String, _topic: Any = None) -> None:
        with self._changed:
            self._records.append(json.loads(msg.data))
            self._changed.notify_all()

    def wait_for(self, match: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        def first() -> dict[str, Any] | None:
            return next((r for r in self._records if match(r)), None)

        with self._changed:
            found = self._changed.wait_for(first, timeout=_TIMEOUT_S)
        assert found is not None, "no matching system_log record"
        return found


@pytest.fixture
def running(tmp_path: Path) -> Iterator[tuple[_Records, LCMTransport[Bool], LCMTransport[String]]]:
    (tmp_path / "main.jsonl").write_text(
        json.dumps({"event": "hello from dimos", "level": "info"}) + "\n"
    )
    module = SystemLog(
        sample_hz=20.0,
        publish_period_s=0.1,
        slow_period_s=1.0,
        log_poll_s=0.05,
        log_dir=str(tmp_path),
    )
    suffix = f"{time.monotonic_ns()}"
    module.system_log.transport = LCMTransport(f"/test_system_log_{suffix}", String)
    module.bad_event.transport = LCMTransport(f"/test_bad_event_{suffix}", Bool)
    module.robot_health.transport = LCMTransport(f"/test_robot_health_{suffix}", String)
    records = _Records()
    listener = LCMTransport(f"/test_system_log_{suffix}", String)
    listener.subscribe(records.add)
    bad_event = LCMTransport(f"/test_bad_event_{suffix}", Bool)
    robot_health = LCMTransport(f"/test_robot_health_{suffix}", String)
    module.start()
    try:
        yield records, bad_event, robot_health
    finally:
        module.stop()
        for transport in (listener, bad_event, robot_health):
            transport.stop()


def test_publishes_health_logs_robot_health_and_dumps_on_bad_event(
    running: tuple[_Records, LCMTransport[Bool], LCMTransport[String]],
) -> None:
    records, bad_event, robot_health = running

    health = records.wait_for(lambda r: r["type"] == "health")
    assert 0.0 <= health["cpu_percent"] <= 100.0
    assert health["memory"]["total_mb"] > 0

    log = records.wait_for(lambda r: r["type"] == "dimos_log")
    assert log["file"] == "main.jsonl"
    assert log["record"] == {"event": "hello from dimos", "level": "info"}

    robot_health.broadcast(None, String(json.dumps({"type": "go2_lowstate", "soc": 55})))
    assert records.wait_for(lambda r: r["type"] == "go2_lowstate")["soc"] == 55

    bad_event.broadcast(None, Bool(True))
    dump = records.wait_for(lambda r: r["type"] == "health_dump")
    assert dump["reason"] == "bad_event"
    assert len(dump["samples"]) > 1
    times = [s["t"] for s in dump["samples"]]
    assert times == sorted(times)
    assert records.wait_for(lambda r: r["type"] == "processes")["top"]
