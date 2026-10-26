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

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from pathlib import Path
import threading
from typing import Any

import pytest
import pytest_mock

from dimos.core.stream import Stream, Transport
from dimos.imitation.collection.episode import EpisodeEvent, EpisodeStatus, ExportStatus
from dimos.imitation.collection.lerobot_export import BUILDING_SUFFIX, LeRobotExportModule
from dimos.imitation.dataprep.schema import DataPrepConfig, OutputConfig
from dimos.msgs.std_msgs.String import String
from dimos.utils.testing.waiting import wait_until


class _DirectTransport(Transport[Any]):
    def __init__(self) -> None:
        self._subscribers: list[Callable[[Any], Any]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._subscribers.clear()

    def broadcast(self, selfstream: Stream[Any] | None, value: Any) -> None:
        for callback in tuple(self._subscribers):
            callback(value)

    def subscribe(
        self, callback: Callable[[Any], Any], selfstream: Stream[Any] | None = None
    ) -> Callable[[], None]:
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)


def _status(event: EpisodeEvent, saved: int) -> String:
    status = EpisodeStatus(
        ts=1.0,
        state="recording" if event == "start" else "idle",
        episodes_saved=saved,
        episodes_discarded=0,
        last_event=event,
        task_label="task",
    )
    return String(status.to_json())


class _Harness:
    """A started exporter with direct status input and captured export statuses."""

    def __init__(self, module: LeRobotExportModule) -> None:
        self.module = module
        self.statuses: list[ExportStatus] = []
        module.status.transport = _DirectTransport()
        module.export_status.subscribe(
            lambda message: self.statuses.append(ExportStatus.from_json(message.data))
        )

    def send(self, event: EpisodeEvent, saved: int) -> None:
        self.module.status.transport.publish(_status(event, saved))

    def wait_for(self, state: str) -> ExportStatus:
        wait_until(
            lambda: bool(self.statuses) and self.statuses[-1].state == state,
            timeout=10.0,
            interval=0.005,
            message=f"export never reached {state}: {[s.state for s in self.statuses]}",
        )
        return self.statuses[-1]


@pytest.fixture
def make_exporter(tmp_path: Path) -> Iterator[Callable[..., _Harness]]:
    config = DataPrepConfig(
        output=OutputConfig(format="lerobot", path=tmp_path / "unused", metadata={"robot": "test"})
    )
    (tmp_path / "config.json").write_text(config.model_dump_json())
    harnesses: list[_Harness] = []

    def make(**overrides: Any) -> _Harness:
        module = LeRobotExportModule(
            source=tmp_path / "session.db",
            dataprep_config=tmp_path / "config.json",
            dataset_path=tmp_path / "dataset",
            **overrides,
        )
        harness = _Harness(module)
        harnesses.append(harness)
        module.start()
        return harness

    yield make
    for harness in harnesses:
        harness.module.stop()


def _fake_dataprep(
    calls: list[DataPrepConfig], gate: threading.Event | None = None
) -> Callable[..., Path]:
    """Stand in for run_dataprep: record the config and write a minimal dataset."""

    def run(config: DataPrepConfig) -> Path:
        calls.append(config)
        if gate is not None and len(calls) == 1:
            assert gate.wait(timeout=10.0)
        root = Path(config.output.path)
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(calls)}))
        return root

    return run


def test_saved_episode_rebuilds_the_dataset_and_reports_it(
    tmp_path: Path, make_exporter: Callable[..., _Harness], mocker: pytest_mock.MockerFixture
) -> None:
    calls: list[DataPrepConfig] = []
    mocker.patch(
        "dimos.imitation.collection.lerobot_export.run_dataprep", side_effect=_fake_dataprep(calls)
    )
    harness = make_exporter()

    harness.send("save", 1)
    done = harness.wait_for("done")

    assert [status.state for status in harness.statuses] == ["idle", "exporting", "done"]
    assert done.episodes == 1
    assert done.dataset == str(tmp_path / "dataset")
    assert (tmp_path / "dataset" / "meta" / "info.json").is_file()
    assert not (tmp_path / f"dataset{BUILDING_SUFFIX}").exists()
    [config] = calls
    assert config.source == str(tmp_path / "session.db")
    assert config.output.format == "lerobot"
    assert config.output.path == tmp_path / f"dataset{BUILDING_SUFFIX}"
    assert config.output.metadata["robot"] == "test"


def test_export_failure_is_reported_and_keeps_the_previous_dataset(
    tmp_path: Path, make_exporter: Callable[..., _Harness], mocker: pytest_mock.MockerFixture
) -> None:
    previous = tmp_path / "dataset" / "meta" / "info.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({"total_episodes": 1}))
    mocker.patch(
        "dimos.imitation.collection.lerobot_export.run_dataprep",
        side_effect=RuntimeError("All saved episodes failed dataset validation"),
    )
    mocker.patch("dimos.imitation.collection.lerobot_export.EXPORT_READY_TIMEOUT_S", 0.0)
    harness = make_exporter()

    harness.send("save", 2)
    failed = harness.wait_for("failed")

    assert failed.error is not None
    assert "failed dataset validation" in failed.error
    assert failed.episodes == 0
    assert previous.is_file()
    assert not (tmp_path / f"dataset{BUILDING_SUFFIX}").exists()


def test_saves_during_an_export_coalesce_into_one_rerun(
    make_exporter: Callable[..., _Harness], mocker: pytest_mock.MockerFixture
) -> None:
    calls: list[DataPrepConfig] = []
    gate = threading.Event()
    mocker.patch(
        "dimos.imitation.collection.lerobot_export.run_dataprep",
        side_effect=_fake_dataprep(calls, gate),
    )
    harness = make_exporter()

    harness.send("save", 1)
    wait_until(lambda: len(calls) == 1, timeout=10.0, interval=0.005)
    harness.send("save", 2)
    harness.send("save", 3)
    gate.set()
    done = harness.wait_for("done")
    wait_until(lambda: harness.module._thread is None, timeout=10.0, interval=0.005)

    assert len(calls) == 2
    assert done.episodes == 2


def test_export_on_save_disabled_exports_once_at_stop(
    make_exporter: Callable[..., _Harness], mocker: pytest_mock.MockerFixture
) -> None:
    calls: list[DataPrepConfig] = []
    mocker.patch(
        "dimos.imitation.collection.lerobot_export.run_dataprep", side_effect=_fake_dataprep(calls)
    )
    harness = make_exporter(export_on_save=False)

    harness.send("save", 1)
    harness.send("save", 2)
    assert calls == []

    harness.module.stop()

    assert len(calls) == 1
    assert harness.statuses[-1].state == "done"
    assert harness.statuses[-1].episodes == 1


def test_events_without_saved_episodes_never_export(
    make_exporter: Callable[..., _Harness], mocker: pytest_mock.MockerFixture
) -> None:
    run = mocker.patch("dimos.imitation.collection.lerobot_export.run_dataprep")
    harness = make_exporter()

    harness.send("start", 0)
    harness.send("discard", 0)
    harness.module.stop()

    run.assert_not_called()
    assert [status.state for status in harness.statuses] == ["idle"]
