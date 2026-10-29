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

"""The walking G1 in sim2 with its head Mid-360."""

import math

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.protocol.tf.static_tf_publisher import StaticTfPublisher
from dimos.robot.unitree.g1.blueprints.primitive.unitree_g1_vis import unitree_g1_vis
from dimos.robot.unitree.g1.g1_tf_publisher import (
    _TORSO_MID360_XYZ,
    MID360_PITCH,
    base_to_torso,
    torso_to_mid360,
)
from dimos.robot.unitree.g1.sim2 import G1Sim
from dimos.sim2.lidar import SimLidar
from dimos.sim2.module import Simulation
from dimos.sim2.spec import Mount


class G1Mid360Mount(StaticTfPublisher):
    """base_link -> torso_link -> mid360_link with the waist at rest."""

    def transforms(self) -> list[Transform]:
        return [base_to_torso(0.0, 0.0, 0.0), torso_to_mid360()]


unitree_g1_sim2 = Simulation.autoconnect(
    unitree_g1_vis,
    G1Sim.blueprint(),
    G1Mid360Mount.blueprint(),
    # Mounted upside down, as torso_to_mid360 has it.
    SimLidar.blueprint(
        robot_id="g1", mount=Mount("torso_link", _TORSO_MID360_XYZ, (math.pi, MID360_PITCH, 0.0))
    ),
).global_config(n_workers=4, robot_model="unitree_g1")
