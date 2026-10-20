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

from dataclasses import replace
import os

from dimos.control.components import HardwareComponent
from dimos.control.coordinator import TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import JOINT_TRAJECTORY_TASK_NAME
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.visualization.config import ManipulationVisualizationConfig
from dimos.manipulation.visualization.viser.config import ViserVisualizationConfig
from dimos.robot.manipulators.dual_openyam.blueprints.basic import (
    DualOpenYamCoordinator,
    DualOpenYamCoordinatorConfig,
)
from dimos.robot.manipulators.dual_openyam.config import dual_openyam_model_config
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_GRIPPER_JOINTS,
)
from dimos.teleop.yam_leader.teleop_module import YamLeaderTeleopModule

# Per-joint follower speed caps for streamed leader targets. Leader-follower
# needs human arm speeds: the OpenArm leader runs 3 rad/s on the arm and 6 on
# the wrists. The YAM arm joints get 3 (the URDF's 2 felt slow next to the
# wrists on the rig, 2026-10-20) and the wrists 4.
DUAL_OPENYAM_LEADER_VELOCITY_RAD_S = 3.0
DUAL_OPENYAM_LEADER_WRIST_VELOCITY_RAD_S = 4.0
_WRIST_JOINT_INDICES = (3, 4, 5)  # joints 4 to 6 of each arm


def dual_openyam_leader_velocity_limits() -> dict[str, float]:
    """Arm joints at the URDF cap, wrist joints at the leader wrist cap."""
    return {
        joint: (
            DUAL_OPENYAM_LEADER_WRIST_VELOCITY_RAD_S
            if index % 6 in _WRIST_JOINT_INDICES
            else DUAL_OPENYAM_LEADER_VELOCITY_RAD_S
        )
        for index, joint in enumerate(DUAL_OPENYAM_ARM_JOINTS)
    }


def with_wrist_gains(
    component: HardwareComponent, kp: float | None, kd: float | None
) -> HardwareComponent:
    """The component with joints 4 to 6 of each arm at the given MIT gains.

    main drives the wrist motors at kp 10, kd 1.5, which follows a leader late;
    the step tests of 2026-10-16 settled on kp 80, kd 1.5 for both arms.
    """
    if kp is None and kd is None:
        return component
    assert component.wb_config is not None
    assert component.wb_config.kp is not None and component.wb_config.kd is not None
    new_kp = list(component.wb_config.kp)
    new_kd = list(component.wb_config.kd)
    for index, joint in enumerate(component.joints):
        if (
            joint in DUAL_OPENYAM_ARM_JOINTS
            and DUAL_OPENYAM_ARM_JOINTS.index(joint) % 6 in _WRIST_JOINT_INDICES
        ):
            if kp is not None:
                new_kp[index] = kp
            if kd is not None:
                new_kd[index] = kd
    return replace(
        component, wb_config=replace(component.wb_config, kp=tuple(new_kp), kd=tuple(new_kd))
    )


class DualOpenYamLeaderCoordinatorConfig(DualOpenYamCoordinatorConfig):
    """Dual OpenYAM coordinator config with optional wrist gain overrides."""

    wrist_kp: float | None = None
    wrist_kd: float | None = None


class DualOpenYamLeaderCoordinator(DualOpenYamCoordinator):
    """DualOpenYamCoordinator whose wrist motor gains can be set per run."""

    config: DualOpenYamLeaderCoordinatorConfig

    def _build_hardware(self) -> HardwareComponent:
        return with_wrist_gains(
            super()._build_hardware(), self.config.wrist_kp, self.config.wrist_kd
        )


# The robot's zenoh router for the split deployment. Named, not scouted.
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
        DualOpenYamLeaderCoordinator.blueprint(
            instance_name="ControlCoordinator",
            tasks=dual_openyam_leader_tasks(),
        ),
        ManipulationModule.blueprint(
            model=dual_openyam_model_config(),
            visualization=visualization,
        ),
    )


# One machine: leaders, coordinator and Viser together. The leader ports given
# select the sides; the follower is in-memory unless both CAN ports are set.
teleop_dual_openyam_leader = autoconnect(
    YamLeaderTeleopModule.blueprint(),
    _follower({"backend": "viser"}),
)

# Split deployment: leaders on an operator machine, follower on the robot. Both
# halves name the streams the same, so they ride the robot's router once the
# operator machine joins it (--robot-ip <robot>). The robot owns the bus-wide
# Coordinator name, so the operator half does not claim it.
teleop_dual_openyam_leader_operator = YamLeaderTeleopModule.blueprint().global_config(
    zenoh_mode="client", serve_coordinator_rpc=False
)

teleop_dual_openyam_leader_follower = _follower(
    ViserVisualizationConfig(host="0.0.0.0")
).global_config(zenoh_connect=DUAL_OPENYAM_ROUTER)
