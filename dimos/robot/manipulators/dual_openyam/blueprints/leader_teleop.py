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

"""Leader-follower teleop of the dual OpenYAM from two YAM leader arms."""

from __future__ import annotations

import os

from dimos.control.coordinator import TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import JOINT_TRAJECTORY_TASK_NAME
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.visualization.config import ManipulationVisualizationConfig
from dimos.manipulation.visualization.viser.config import ViserVisualizationConfig
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.config import (
    DUAL_OPENYAM_WRIST_JOINT_INDICES,
    dual_openyam_model_config,
)
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_GRIPPER_JOINTS,
)
from dimos.teleop.yam_leader.teleop_module import YamLeaderTeleopModule

# Leader-follower needs human arm speeds; the URDF's 2 rad/s felt slow on the rig.
DUAL_OPENYAM_LEADER_VELOCITY_RAD_S = 3.0
DUAL_OPENYAM_LEADER_WRIST_VELOCITY_RAD_S = 4.0


def dual_openyam_leader_velocity_limits() -> dict[str, float]:
    """Arm joints at the URDF cap, wrist joints at the leader wrist cap."""
    return {
        joint: (
            DUAL_OPENYAM_LEADER_WRIST_VELOCITY_RAD_S
            if index % 6 in DUAL_OPENYAM_WRIST_JOINT_INDICES
            else DUAL_OPENYAM_LEADER_VELOCITY_RAD_S
        )
        for index, joint in enumerate(DUAL_OPENYAM_ARM_JOINTS)
    }


# Named, not scouted.
DUAL_OPENYAM_ROUTER = os.environ.get("DIMOS_OPENYAM_ROUTER", "tcp/127.0.0.1:7447")


def dual_openyam_leader_tasks() -> list[TaskConfig]:
    """Streamed joint targets for both arms plus one gripper task per side."""
    return [
        TaskConfig(
            name=JOINT_TRAJECTORY_TASK_NAME,
            type="trajectory",
            joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
            priority=10,
            params={
                "start_position_tolerance": 0.05,
                "velocity_limits": dual_openyam_leader_velocity_limits(),
            },
        ),
        TaskConfig(
            name="left_arm_gripper",
            type="gripper",
            joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[0]],
            priority=20,
            stream_bind={"gripper_command": "left_gripper_command"},
        ),
        TaskConfig(
            name="right_arm_gripper",
            type="gripper",
            joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[1]],
            priority=20,
            stream_bind={"gripper_command": "right_gripper_command"},
        ),
    ]


def _follower(visualization: ManipulationVisualizationConfig | dict[str, str]) -> Blueprint:
    return autoconnect(
        DualOpenYamCoordinator.blueprint(
            instance_name="ControlCoordinator",
            tasks=dual_openyam_leader_tasks(),
        ),
        ManipulationModule.blueprint(
            model=dual_openyam_model_config(),
            visualization=visualization,
        ),
    )


# The leader ports given select the sides; the follower is in-memory unless both CAN ports are set.
teleop_dual_openyam_leader = autoconnect(
    YamLeaderTeleopModule.blueprint(),
    _follower({"backend": "viser"}),
)

# Leaders on the operator machine (--robot-ip <robot>); the robot owns the Coordinator name.
teleop_dual_openyam_leader_operator = YamLeaderTeleopModule.blueprint().global_config(
    zenoh_mode="client", serve_coordinator_rpc=False
)

teleop_dual_openyam_leader_follower = _follower(
    ViserVisualizationConfig(host="0.0.0.0")
).global_config(zenoh_connect=DUAL_OPENYAM_ROUTER)
