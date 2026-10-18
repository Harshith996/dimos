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

"""Menagerie Go2 devices with the physical SF Mid360 mount. No policy runtime."""

import math

from dimos.control.go2_freewalk.policy import JOINT_NAMES
from dimos.robot.unitree.go2.go2_mid360_static_transforms import CAMERA_XYZ, MID360_XYZ
from dimos.sim2.sensors.lidar.models.fibonacci import Fibonacci
from dimos.sim2.sensors.lidar.models.mid360 import Mid360
from dimos.sim2.sensors.spec import Camera, Imu, Lidar, Mount
from dimos.sim2.spec import ControlInterface, Joint, RobotConfig
from dimos.utils.data import LfsPath

MID360_MOUNT = Mount(
    "base",
    xyz=(
        CAMERA_XYZ[0] + MID360_XYZ[0],
        CAMERA_XYZ[1] + MID360_XYZ[1],
        CAMERA_XYZ[2] + MID360_XYZ[2],
    ),
    rpy=(0, math.radians(60), 0),
)

GO2_FREEWALK = RobotConfig(
    model=LfsPath("go2_menagerie/go2.xml"),
    root_body="base",
    floating=True,
    spawn_height=0.33,
    control=ControlInterface.WHOLE_BODY,
    joints=tuple(
        Joint(
            name=name, model_name=name, actuator=name.removesuffix("_joint"), home=home, kp=40, kd=1
        )
        for name, home in zip(
            JOINT_NAMES,
            (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5),
            strict=True,
        )
    ),
    sensors=(
        Imu("imu", "imu", rate_hz=200),
        Lidar("lidar", MID360_MOUNT, Fibonacci, self_occlusion=True),
        Camera("camera", Mount("base", CAMERA_XYZ, (math.pi / 2, 0, -math.pi / 2)), fovy=60),
    ),
)

GO2_FREEWALK_MID360 = GO2_FREEWALK.with_sensor(
    Lidar("lidar", MID360_MOUNT, Mid360, self_occlusion=True, imu=Imu("lidar_imu", MID360_MOUNT))
)
