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

"""Standalone SONIC with the public G1Connection, on hardware or sim2."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_groot_wbc import _nav_stack, _viewer
from dimos.robot.unitree.g1.connection import G1Connection
from dimos.robot.unitree.g1.control_safety import COMMAND_TIMEOUT_SECONDS
from dimos.robot.unitree.g1.motor_device import G1MotorConfig
from dimos.robot.unitree.g1.sonic_policy import G1SonicPolicyModule

_simulated = global_config.simulation == "mujoco"

_unitree_g1_sonic_wbc_core = (
    autoconnect(
        G1Connection.blueprint(
            controller="sonic",
            motors=G1MotorConfig(command_timeout_seconds=COMMAND_TIMEOUT_SECONDS),
        ),
        G1SonicPolicyModule.blueprint(
            cpu_simulation=global_config.simulation == "mujoco",
            auto_arm=_simulated,
            auto_dry_run=not _simulated,
            default_ramp_seconds=0.0 if _simulated else 3.0,
        ),
    )
    .remappings(
        [
            (G1Connection, "imu", "low_level_imu"),
            ("g1connection_lidar", "pointcloud", "pointcloud"),
            ("g1connection_camera", "color_image", "color_image"),
            ("g1connection_camera", "camera_info", "camera_info"),
            ("g1connection_camera", "depth_image", "depth_image"),
            ("g1connection_camera", "depth_camera_info", "depth_camera_info"),
        ]
    )
    .global_config(robot_model="unitree_g1")
)

unitree_g1_sonic_wbc = autoconnect(_unitree_g1_sonic_wbc_core, _nav_stack, _viewer())
