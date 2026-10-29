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

"""An IMU read from MuJoCo's gyro, accelerometer and orientation sensors at a site."""

import asyncio
from collections.abc import AsyncIterator
import time

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.sim2.module import SimulationSpec
from dimos.sim2.spec import Mount, SimulatedSensor
from dimos.sim2.state import SimClient


class SimImuConfig(ModuleConfig):
    name: str = "imu"
    mount: Mount = Mount("world")
    robot_id: str | None = None
    frame_id: str | None = "imu_link"
    rate_hz: float = 200.0


class SimImu(Module):
    config: SimImuConfig
    imu: Out[Imu]
    _simulation: SimulationSpec

    @staticmethod
    def describe_simulated(config: SimImuConfig) -> SimulatedSensor:
        return SimulatedSensor(config.name, "imu", config.mount, config.robot_id)

    async def main(self) -> AsyncIterator[None]:
        self._reading = asyncio.create_task(self._read(SimClient(self._simulation.describe())))
        yield

    async def _read(self, client: SimClient) -> None:
        sensor = f"{self.config.robot_id or 'world'}/{self.config.name}"
        while True:
            client.state.read(client.data)
            w, x, y, z = client.data.sensor(f"{sensor}/orientation").data
            self.imu.publish(
                Imu(
                    angular_velocity=Vector3(*client.data.sensor(f"{sensor}/gyro").data),
                    linear_acceleration=Vector3(
                        *client.data.sensor(f"{sensor}/accelerometer").data
                    ),
                    orientation=Quaternion(x, y, z, w),
                    frame_id=self.frame_id,
                    ts=time.time(),
                )
            )
            await asyncio.sleep(1 / self.config.rate_hz)
