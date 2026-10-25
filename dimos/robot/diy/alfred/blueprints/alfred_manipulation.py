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

    dimos run alfred-manipulation              # top-down heuristic grasps
    dimos run alfred-manipulation-graspgenx    # GraspGenX grasps (its own env, synced on first start)

The Orin NX runs one of two stacks at a time: this one for manipulation, ``alfred-navigation``
for driving around. Nothing of navigation is here: no Mid-360, no Point-LIO, no planners, no
follower. The base still drives from the Controller (teleop is a claim on the coordinator), and
a whole-body plan's base segment is a motion from where the robot stands, tracked on the
FlowBase's own wheel odometry (``WheelOdometryRelay`` puts it on ``/flowbase/odom`` and tf).
The D455 feeds the planner's obstacle map and, with a GPU, the perception stack's localize
(``localize_boxes``): the Manipulation app's scan.
"""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.diy.alfred.blueprints.alfred_full import (
    BASE_FRAME,
    ODOM_FRAME,
    alfred_arm_modules,
    alfred_base_transports,
    alfred_coordinator,
)
from dimos.robot.diy.alfred.blueprints.alfred_full_perception import (
    GraspGenerator,
    alfred_perception_modules,
)
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevel
from dimos.robot.diy.alfred.mount_tf import AlfredMountTf
from dimos.robot.diy.alfred.wheel_odometry_relay import WheelOdometryRelay


def alfred_bench_modules(*, grasp: GraspGenerator = "heuristic") -> tuple[Any, ...]:
    """The bench stack's modules, with one grasp generator answering the pick pipeline."""
    return (
        # The mounts rooted at base_link: no lidar owns a frame above it here.
        AlfredMountTf.blueprint(),
        AlfredHighLevel.blueprint(),
        WheelOdometryRelay.blueprint(world_frame=ODOM_FRAME, base_frame=BASE_FRAME),
        # No Rerun bridge: the Desktop reads zenoh itself, and the bridge's late-viewer buffer
        # is what grew to 6.4 GB and got this stack OOM-killed (2026-10-25).
        *alfred_arm_modules(),
        alfred_coordinator(arms=True, follower=False),
        # The pick pipeline's scene registration only: localize's second OWLv2 + EdgeTAM and
        # its frame memory pushed the Orin into swap beside it.
        *alfred_perception_modules(localize=False, grasp=grasp),
    )


# The registry scans for module-level autoconnect(...) chains: each stack is spelled out.
alfred_manipulation = (
    autoconnect(*alfred_bench_modules())
    .transports(alfred_base_transports())
    .global_config(n_workers=10, robot_model="alfred")
)
# The same stack with GraspGenX proposing grasps instead of the top-down heuristic. GraspGenX
# runs in its own environment, synced on first start (its weights download then too).
alfred_manipulation_graspgenx = (
    autoconnect(*alfred_bench_modules(grasp="graspgenx"))
    .transports(alfred_base_transports())
    .global_config(n_workers=10, robot_model="alfred")
)
