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

"""A Mid-360 cast into the running simulation with the fitted scan pattern and return model."""

import asyncio
from collections.abc import AsyncIterator
import time

import numpy as np

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.sim2.module import SimulationSpec
from dimos.sim2.spec import Mount, SimulatedSensor
from dimos.sim2.state import SimClient
from dimos.simulation.sensors.mid360.lidar import OcclusionMap, SimMid360
from dimos.simulation.sensors.mid360.pattern import POINT_RATE, Mid360Pattern
from dimos.simulation.sensors.mujoco_raycaster import MujocoRaycaster
from dimos.utils.data import get_data

GEOM_GROUPS = 6


class SimLidarConfig(ModuleConfig):
    name: str = "lidar"
    mount: Mount = Mount("world")
    robot_id: str | None = None
    frame_id: str | None = "mid360_link"
    rate_hz: float = 10.0
    # Robots' meshes sit outside group 0, so the default sees only the scene, and stays cheap.
    geom_groups: tuple[int, ...] = (0,)
    # The Go2 rig's mount blocks some directions; only meaningful when mounted on a Go2.
    go2_occlusion: bool = False
    seed: int = 0


class SimLidar(Module):
    config: SimLidarConfig
    lidar: Out[PointCloud2]
    _simulation: SimulationSpec

    @staticmethod
    def describe_simulated(config: SimLidarConfig) -> SimulatedSensor:
        return SimulatedSensor(config.name, "lidar", config.mount, config.robot_id)

    async def main(self) -> AsyncIterator[None]:
        client = SimClient(self._simulation.describe())
        root = get_data("go2_sim")
        sensor = SimMid360(
            MujocoRaycaster(
                client.model,
                client.data,
                np.isin(np.arange(GEOM_GROUPS), self.config.geom_groups).astype(np.uint8),
            ),
            Mid360Pattern.load(root / "mid360_pattern.npz"),
            self.config.seed,
            OcclusionMap.load(root / "go2_mid360_occlusion.npz")
            if self.config.go2_occlusion
            else None,
        )
        self._scanning = asyncio.create_task(self._scan(client, sensor))
        yield

    async def _scan(self, client: SimClient, sensor: SimMid360) -> None:
        site = f"{self.config.robot_id or 'world'}/{self.config.name}"
        points_per_scan = int(POINT_RATE / self.config.rate_hz)
        deadline = time.monotonic()
        while True:
            data = client.read()
            pose = data.site(site)
            points = sensor.cast(pose.xpos, pose.xmat.reshape(3, 3), points_per_scan)
            self.lidar.publish(PointCloud2.from_numpy(points, self.frame_id, time.time()))
            deadline = max(deadline + 1 / self.config.rate_hz, time.monotonic())
            await asyncio.sleep(deadline - time.monotonic())
