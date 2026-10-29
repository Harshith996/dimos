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

"""An RGB camera rendered from the running simulation."""

import asyncio
from collections.abc import AsyncIterator
import time

import mujoco

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.sim2.module import SimulationSpec
from dimos.sim2.spec import Mount, SimulatedSensor
from dimos.sim2.state import SimClient


class SimCameraConfig(ModuleConfig):
    name: str = "camera"
    mount: Mount = Mount("world")
    robot_id: str | None = None
    frame_id: str | None = "camera_optical"
    width: int = 640
    height: int = 480
    fovy: float = 60.0
    rate_hz: float = 10.0


class SimCamera(Module):
    config: SimCameraConfig
    color_image: Out[Image]
    _simulation: SimulationSpec

    @staticmethod
    def describe_simulated(config: SimCameraConfig) -> SimulatedSensor:
        return SimulatedSensor(config.name, "camera", config.mount, config.robot_id, config.fovy)

    async def main(self) -> AsyncIterator[None]:
        client = SimClient(self._simulation.describe())
        renderer = mujoco.Renderer(client.model, self.config.height, self.config.width)
        self._capturing = asyncio.create_task(self._capture(client, renderer))
        yield
        self._capturing.cancel()
        renderer.close()

    async def _capture(self, client: SimClient, renderer: mujoco.Renderer) -> None:
        camera = f"{self.config.robot_id or 'world'}/{self.config.name}"
        while True:
            renderer.update_scene(client.read(), camera=camera)
            rgb = renderer.render().copy()
            self.color_image.publish(
                Image(data=rgb, format=ImageFormat.RGB, frame_id=self.frame_id, ts=time.time())
            )
            await asyncio.sleep(1 / self.config.rate_hz)
