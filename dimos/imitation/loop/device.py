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

"""The example's device boundary; policies see joint feedback and camera RGB only."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from dimos.robot.manipulators.openyam.control import YamControl
from dimos.robot.manipulators.openyam.device import AcceptedTarget, YamPositionDevice
from dimos.robot.manipulators.openyam.simulation import YamArtifact
from dimos.robot.manipulators.openyam.workcell import FPS, learning_world
from dimos.sim2.messages import StreamDescriptor
from dimos.sim2.module import SimulationModule
from dimos.sim2.sensors.camera.device import CameraDevice
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.sensors.reader import WorldReader


@dataclass(frozen=True)
class Observation:
    state: NDArray[np.float64]
    state_ts: float
    image: NDArray[np.uint8]
    image_ts: float
    generation: int
    applied_sequence: int
    accepted: AcceptedTarget | None = None
    received_at: float | None = None


class OpenYamSimulation:
    """Own the existing sim2 module and one public writable device connection.

    ControlCoordinator owns writes through YamPositionDevice. The learning session
    observes feedback and submits targets; it never owns a writable client.
    """

    def __init__(self, artifact: YamArtifact, evidence: Path) -> None:
        self.artifact = artifact
        self.world = learning_world(artifact)
        self.module = SimulationModule(
            world=self.world, sim_id=f"learn-{uuid4().hex}", evidence_dir=evidence
        )
        self._stack = ExitStack()
        self._images: dict[str, tuple[NDArray[np.uint8], float, int]] = {}
        self.cameras: list[CameraDevice] = []
        self.control: YamControl | None = None
        self.device: YamPositionDevice | None = None

    def __enter__(self) -> OpenYamSimulation:
        try:
            self._stack.callback(self.module.stop)
            self.module.start()
            description = self.module.describe()
            self.device = YamPositionDevice(
                StreamDescriptor.from_dict(description["robots"]["openyam"]), self.artifact
            )
            self.control = YamControl(self.device)
            self._stack.callback(self.control.close)
            self.control.start()
            for robot, sensor in self.world.devices:
                if not isinstance(sensor, Camera):
                    continue
                reader = WorldReader(description)
                self._stack.callback(reader.close)
                camera = CameraDevice(reader, sensor, robot)
                self._stack.callback(camera.close)
                self.cameras.append(camera)
            return self
        except BaseException:
            self._stack.close()
            raise

    def observe(self) -> Observation | None:
        for camera in self.cameras:
            for sample in camera.poll():
                if sample.port == "color_image":
                    self._images[camera.sensor.name] = (
                        sample.value.data.copy(),
                        sample.value.ts,
                        sample.episode,
                    )
                    if camera.sensor.name == "wrist" and self.control is not None:
                        self.control.publish_camera(sample.value)
        assert self.device is not None
        frame, accepted = self.device.snapshot()
        image = self._images.get("wrist")
        if image is None or image[2] != frame.metadata.episode_id:
            return None
        state_ts = float(frame.values["wall_time"][0])
        if abs(state_ts - image[1]) > 0.1 or time.time() - state_ts > 0.5:
            return None
        return Observation(
            frame.values["position"].copy(),
            state_ts,
            image[0],
            image[1],
            frame.metadata.episode_id,
            frame.metadata.applied_action_sequence,
            accepted,
            time.time(),
        )

    @property
    def overview(self) -> NDArray[np.uint8] | None:
        sample = self._images.get("overview")
        return sample[0] if sample else None

    def command(self, action: NDArray[np.float64], timestamp: float) -> None:
        assert self.control is not None
        if time.time() - timestamp > 1 / FPS:
            raise ValueError("target expired before coordinator dispatch")
        self.control.target(action.tolist())

    def __exit__(self, *_: Any) -> None:
        self._stack.close()
