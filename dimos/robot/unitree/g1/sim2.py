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

"""G1 device model and sensor mounts; controller selection does not change the robot."""

from dataclasses import replace
from pathlib import Path

from dimos.robot.unitree.g1.control_config import (
    G1_GROOT_HOME,
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_joints,
)
from dimos.sim2.sensors.lidar.models.fibonacci import Fibonacci
from dimos.sim2.sensors.spec import Camera, Imu, Lidar
from dimos.sim2.spec import ControlInterface, Joint, RobotConfig
from dimos.utils.data import LfsPath

G1_GROOT = RobotConfig(
    model=Path(__file__).parent / "assets" / "g1_29dof.xml",
    meshdir=LfsPath("g1_urdf/meshes"),
    root_body="pelvis",
    floating=True,
    spawn_height=0.793,
    wait_for_first_command=True,
    control=ControlInterface.WHOLE_BODY,
    joints=tuple(
        Joint(
            name=name,
            model_name=name.split("/", 1)[1] + "_joint",
            actuator=name.split("/", 1)[1] + "_joint",
            home=home,
            kp=kp,
            kd=kd,
        )
        for name, home, kp, kd in zip(
            g1_joints, G1_GROOT_HOME, G1_GROOT_KP, G1_GROOT_KD, strict=True
        )
    ),
    sensors=(
        Imu("imu", site="control_imu"),
        Lidar("lidar", "mid360_link", Fibonacci, rate_hz=10.0, maximum_world_elevation=0.0),
        Camera("camera", camera="front_camera"),
    ),
)


def g1_definition(controller: str, source: str = "g1") -> RobotConfig:
    """Bind the selected controller's home/gains, preserving its trained joint order."""
    definition = G1_GROOT
    if controller == "sonic":
        # SONIC inference is optional; importing a robot definition does not load it.
        from dimos.control.sonic.sonic_pipeline import DEFAULT_ANGLES_DDS, SONIC_KD, SONIC_KP

        definition = replace(
            definition,
            joints=tuple(
                replace(joint, home=float(q), kp=float(kp), kd=float(kd))
                for joint, q, kp, kd in zip(
                    definition.joints, DEFAULT_ANGLES_DDS, SONIC_KP, SONIC_KD, strict=True
                )
            ),
        )
    elif controller != "groot":
        raise ValueError("MuJoCo requires groot or sonic; stock Unitree firmware is not emulated")
    return replace(
        definition,
        joints=tuple(
            replace(joint, name=f"{source}/{joint.name.split('/', 1)[1]}")
            for joint in definition.joints
        ),
    )
