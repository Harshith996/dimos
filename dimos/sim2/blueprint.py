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

"""Compose one world and its sensors; robot connections own their device clients."""

from pathlib import Path

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.sim2.module import SimulationModule
from dimos.sim2.sensors.camera.module import SimCameraModule, SimRGBDCameraModule
from dimos.sim2.sensors.lidar.module import LidarModule
from dimos.sim2.sensors.spec import Camera, Imu, Lidar
from dimos.sim2.spec import RobotInstance, WorldConfig


def simulation(
    *,
    scene: Path,
    robots: dict[str, RobotInstance],
    sim_id: str = "sim",
    viewer: bool = False,
    timestep: float = 0.005,
) -> Blueprint:
    """Build physics and declared sensing, without choosing a controller or robot driver."""
    modules = [
        SimulationModule.blueprint(
            world=WorldConfig(scene=scene, robots=robots, timestep=timestep),
            sim_id=sim_id,
            viewer=viewer,
        )
    ]
    ports: tuple[str, ...]
    for robot_id, instance in robots.items():
        config = instance.config
        for sensor in config.sensors:
            kwargs = dict(
                instance_name=f"{robot_id}_{sensor.name}",
                robot_id=robot_id,
                sensor=sensor,
                rate_hz=sensor.rate_hz,
            )
            if isinstance(sensor, Camera):
                module = SimRGBDCameraModule if sensor.depth else SimCameraModule
                blueprint = module.blueprint(**kwargs)
                ports = ("color_image", "camera_info")
                if sensor.depth:
                    ports += ("depth_image", "depth_camera_info")
            elif isinstance(sensor, Lidar):
                if sensor.imu is not None:
                    kwargs["rate_hz"] = max(sensor.rate_hz, sensor.imu.rate_hz)
                blueprint = LidarModule.blueprint(**kwargs, root_body=config.root_body)
                ports = ("pointcloud", "raw_pointcloud", "imu_raw")
            elif isinstance(sensor, Imu):
                # The robot client reads this IMU in the same frame as its joint feedback.
                continue
            else:
                raise TypeError(f"unsupported sensor configuration: {type(sensor).__name__}")
            if len(robots) > 1 or sum(type(s) is type(sensor) for s in config.sensors) > 1:
                blueprint = blueprint.remappings(
                    [
                        (f"{robot_id}_{sensor.name}", port, f"{robot_id}/{sensor.name}/{port}")
                        for port in ports
                    ]
                )
            modules.append(blueprint)
    return autoconnect(*modules)
