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

"""Dual OpenYAM episode recording and its LeRobot export."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dimos.core.stream import Stream, Transport
from dimos.imitation.collection.episode import (
    EpisodeEvent,
    EpisodeStatus,
    ExportStatus,
    RecordingState,
)
from dimos.imitation.collection.lerobot_export import LeRobotExportModule
from dimos.imitation.dataprep.build import inspect_dataset
from dimos.imitation.dataprep.schema import DataPrepConfig
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.manipulators.dual_openyam.collection import (
    DUAL_OPENYAM_CAMERA_STREAMS,
    DUAL_OPENYAM_COLLECTION_STREAMS,
    LEROBOT_DATAPREP_CONFIG,
    DualOpenYamCollectionRecorder,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS
from dimos.teleop.webxr.camera import TeleopCameraConfig
from dimos.utils.testing.waiting import wait_until

TASK = "stack the cups"


def _config() -> DataPrepConfig:
    return DataPrepConfig.model_validate_json(LEROBOT_DATAPREP_CONFIG.read_text())


def _recorder_ports() -> set[str]:
    [atom] = DualOpenYamCollectionRecorder.blueprint().blueprints
    return {stream.name for stream in atom.streams if stream.direction == "in"} - {"tf"}


def test_lerobot_config_matches_the_recorder_and_the_robot() -> None:
    config = _config()
    features = {**config.observation, **config.action}
    cameras = {key: spec for key, spec in config.observation.items() if spec.dtype == "video"}
    capture = TeleopCameraConfig()

    assert _recorder_ports() == set(DUAL_OPENYAM_COLLECTION_STREAMS)
    assert {spec.stream for spec in features.values()} <= _recorder_ports()
    assert {spec.stream for spec in cameras.values()} == set(DUAL_OPENYAM_CAMERA_STREAMS)
    assert all(spec.shape == (capture.height, capture.width, 3) for spec in cameras.values())
    assert config.observation["state"].names == DUAL_OPENYAM_JOINTS
    assert config.action["action"].names == DUAL_OPENYAM_JOINTS
    assert config.sync.anchor in cameras
    assert config.episodes.status_stream == "status"
    assert config.output.format == "lerobot"


class _DirectTransport(Transport[Any]):
    """Synchronous in-process transport so the recorder's real port subscriptions run."""

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


def _status(
    ts: float, event: EpisodeEvent, state: RecordingState, saved: int, discarded: int
) -> String:
    status = EpisodeStatus(
        ts=ts,
        state=state,
        episodes_saved=saved,
        episodes_discarded=discarded,
        last_event=event,
        task_label=TASK,
    )
    return String(status.to_json())


@pytest.fixture
def recorder(tmp_path: Path) -> Iterator[DualOpenYamCollectionRecorder]:
    recorder = DualOpenYamCollectionRecorder(
        db_path=tmp_path / "session.db",
        poseless_streams=list(DUAL_OPENYAM_COLLECTION_STREAMS),
    )
    for name in DUAL_OPENYAM_COLLECTION_STREAMS:
        getattr(recorder, name).transport = _DirectTransport()
    recorder.start()
    yield recorder
    recorder.stop()


@pytest.fixture
def exporter(tmp_path: Path) -> Iterator[LeRobotExportModule]:
    exporter = LeRobotExportModule(
        source=tmp_path / "session.db",
        dataprep_config=LEROBOT_DATAPREP_CONFIG,
        dataset_path=tmp_path / "dataset",
    )
    exporter.status.transport = _DirectTransport()
    yield exporter
    exporter.stop()


@pytest.mark.self_hosted
@pytest.mark.skipif_no_turbojpeg
def test_recorded_episodes_export_as_a_lerobot_dataset(
    tmp_path: Path, recorder: DualOpenYamCollectionRecorder, exporter: LeRobotExportModule
) -> None:
    config = _config()
    rate = config.sync.rate_hz
    frames_per_take = 8
    counts = dict.fromkeys(DUAL_OPENYAM_COLLECTION_STREAMS, 0)
    height, width, _ = config.observation["egocentric"].shape

    def publish(name: str, message: Any) -> None:
        counts[name] += 1
        getattr(recorder, name).transport.publish(message)
        wait_until(
            lambda: recorder.store.stream(name).count() == counts[name],
            timeout=5.0,
            interval=0.005,
            message=f"{name} message {counts[name]} was not recorded",
        )

    takes: list[tuple[float, EpisodeEvent]] = [(100.0, "save"), (110.0, "discard"), (120.0, "save")]
    saved = discarded = 0
    for take, (start, outcome) in enumerate(takes):
        publish("status", _status(start, "start", "recording", saved, discarded))
        for frame in range(frames_per_take):
            ts = start + frame / rate
            for camera, stream in enumerate(DUAL_OPENYAM_CAMERA_STREAMS):
                pixel = 40 * take + 10 * camera + frame
                publish(
                    stream,
                    Image(
                        data=np.full((height, width, 3), pixel, dtype=np.uint8),
                        format=ImageFormat.RGB,
                        frame_id=stream,
                        ts=ts,
                    ),
                )
            publish(
                "coordinator_joint_state",
                JointState(
                    ts=ts,
                    frame_id="dual_openyam",
                    name=list(DUAL_OPENYAM_JOINTS),
                    position=[
                        100.0 * take + frame + joint for joint in range(len(DUAL_OPENYAM_JOINTS))
                    ],
                    velocity=[0.0] * len(DUAL_OPENYAM_JOINTS),
                    effort=[0.0] * len(DUAL_OPENYAM_JOINTS),
                ),
            )
        if outcome == "save":
            saved += 1
        else:
            discarded += 1
        publish(
            "status", _status(start + frames_per_take / rate, outcome, "idle", saved, discarded)
        )
    recorder.stop()

    # The exporter rebuilds the dataset from the recording when a save lands.
    exports: list[ExportStatus] = []
    exporter.export_status.subscribe(
        lambda message: exports.append(ExportStatus.from_json(message.data))
    )
    exporter.start()
    exporter.status.transport.publish(_status(takes[-1][0] + 1.0, "save", "idle", saved, discarded))
    wait_until(
        lambda: bool(exports) and exports[-1].state in ("done", "failed"),
        timeout=120.0,
        interval=0.05,
        message="LeRobot export did not finish",
    )
    assert exports[-1].state == "done", exports[-1].error
    assert exports[-1].episodes == 2
    dataset = tmp_path / "dataset"

    info = json.loads((dataset / "meta" / "info.json").read_text())
    image_features = {key for key in info["features"] if key.startswith("observation.images.")}
    assert image_features == {
        "observation.images.egocentric",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    assert info["features"]["observation.state"]["shape"] == [len(DUAL_OPENYAM_JOINTS)]
    assert info["features"]["action"]["shape"] == [len(DUAL_OPENYAM_JOINTS)]
    assert (info["total_episodes"], info["total_frames"], info["fps"]) == (
        2,
        2 * frames_per_take,
        rate,
    )
    assert info["robot_type"] == "dual_openyam"
    for key in image_features:
        assert (dataset / "videos" / key / "chunk-000" / "file-000.mp4").is_file()

    summary = inspect_dataset(dataset)
    assert (summary["episodes"], summary["frames"]) == (2, 2 * frames_per_take)
    meta = json.loads((dataset / "dimos_meta.json").read_text())
    assert [(episode["start_ts"], episode["task_label"]) for episode in meta["episodes"]] == [
        (100.0, TASK),
        (120.0, TASK),
    ]
