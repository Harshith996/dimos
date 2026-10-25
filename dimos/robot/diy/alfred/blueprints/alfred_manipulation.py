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

"""Alfred at the bench: the arms, the lift, the whole-body planner and the mast D455, no lidar.

    dimos run alfred-manipulation

The Orin NX runs one of two stacks at a time: this one for manipulation, ``alfred-navigation``
for driving around. Nothing of navigation is here: no Mid-360, no Point-LIO, no planners, no
follower. The base still drives from the Controller (teleop is a claim on the coordinator), and
a whole-body plan's base segment is a motion from where the robot stands, tracked on the
FlowBase's own wheel odometry (``WheelOdometryRelay`` puts it on ``/flowbase/odom`` and tf).
The D455 feeds the planner's obstacle map and, with a GPU, the perception stack's localize
(``localize_boxes``): the Manipulation app's scan.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.diy.alfred.blueprints.alfred_full import (
    BASE_FRAME,
    ODOM_FRAME,
    _rerun_config,
    alfred_arm_modules,
    alfred_base_transports,
    alfred_coordinator,
)
from dimos.robot.diy.alfred.blueprints.alfred_full_perception import alfred_perception_modules
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevel
from dimos.robot.diy.alfred.mount_tf import AlfredMountTf
from dimos.robot.diy.alfred.wheel_odometry_relay import WheelOdometryRelay
from dimos.visualization.vis_module import vis_module

alfred_manipulation = (
    autoconnect(
        # The mounts rooted at base_link: no lidar owns a frame above it here.
        AlfredMountTf.blueprint(),
        AlfredHighLevel.blueprint(),
        WheelOdometryRelay.blueprint(world_frame=ODOM_FRAME, base_frame=BASE_FRAME),
        vis_module(global_config.viewer, rerun_config=_rerun_config()),
        *alfred_arm_modules(),
        alfred_coordinator(arms=True, follower=False),
        *alfred_perception_modules(),
    )
    .transports(alfred_base_transports())
    .global_config(n_workers=10, robot_model="alfred")
)
