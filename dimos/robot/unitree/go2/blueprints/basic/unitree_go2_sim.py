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

"""Go2 stacks with GO2Connection swapped for a sim2 world, Go2Sim and simulated sensors."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.simulation.model import go2_camera, go2_lidar
from dimos.robot.unitree.go2.simulation.module import Go2Sim
from dimos.sim2.module import Simulation
from dimos.sim2.sensors.camera.module import SimRGBDCameraModule
from dimos.sim2.sensors.lidar.module import LidarModule

_go2_sim = autoconnect(
    Simulation.blueprint(),
    Go2Sim.blueprint(),
    SimRGBDCameraModule.blueprint(robot_id="go2", sensor=go2_camera()),
    LidarModule.blueprint(robot_id="go2", sensor=go2_lidar(), root_body="base"),
).remappings(
    [
        (LidarModule, "pointcloud", "lidar"),
        (SimRGBDCameraModule, "pointcloud", "depth_pointcloud"),
    ]
)

unitree_go2_basic_sim = autoconnect(unitree_go2_basic.disabled_modules(GO2Connection), _go2_sim)

unitree_go2_sim = autoconnect(unitree_go2.disabled_modules(GO2Connection), _go2_sim)
