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

from typing import Any

from dimos.control.coordinator import TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import JOINT_TRAJECTORY_TASK_NAME
from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.blueprints import Blueprint
from dimos.core.coordination.module_coordinator import _is_name_unique
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.dual_openyam.blueprints import leader_teleop
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_GRIPPER_JOINTS,
)
from dimos.teleop.yam_leader.teleop_module import (
    YamLeaderTeleopModule,
    YamLeaderTeleopModuleConfig,
)


def _kwargs(blueprint: Blueprint, module_type: type) -> dict[str, Any]:
    return next(atom.kwargs for atom in blueprint.blueprints if atom.module is module_type)


def _types(blueprint: Blueprint) -> list[type]:
    return [atom.module for atom in blueprint.blueprints]


def _tasks(blueprint: Blueprint) -> list[TaskConfig]:
    tasks = _kwargs(blueprint, DualOpenYamCoordinator)["tasks"]
    assert all(isinstance(task, TaskConfig) for task in tasks)
    return list(tasks)


def test_one_machine_blueprint_streams_both_arms_and_grippers() -> None:
    blueprint = leader_teleop.teleop_dual_openyam_leader
    assert _types(blueprint) == [YamLeaderTeleopModule, DualOpenYamCoordinator, ManipulationModule]
    assert _kwargs(blueprint, DualOpenYamCoordinator)["instance_name"] == "ControlCoordinator"
    for stream in ("joint_command", "left_gripper_command", "right_gripper_command"):
        assert _is_name_unique(blueprint, stream)

    trajectory, left_gripper, right_gripper = _tasks(blueprint)
    assert (trajectory.name, trajectory.type) == (JOINT_TRAJECTORY_TASK_NAME, "trajectory")
    assert trajectory.joint_names == list(DUAL_OPENYAM_ARM_JOINTS)
    assert trajectory.params["velocity_limits"] == dict.fromkeys(
        DUAL_OPENYAM_ARM_JOINTS, leader_teleop.DUAL_OPENYAM_LEADER_VELOCITY_RAD_S
    )
    assert trajectory.priority < left_gripper.priority
    assert (left_gripper.type, left_gripper.joint_names) == (
        "gripper",
        [DUAL_OPENYAM_GRIPPER_JOINTS[0]],
    )
    assert left_gripper.stream_bind == {"gripper_command": "left_gripper_command"}
    assert right_gripper.stream_bind == {"gripper_command": "right_gripper_command"}
    assert _kwargs(blueprint, ManipulationModule)["visualization"] == {"backend": "viser"}


def test_leader_ports_select_the_sides_from_the_command_line() -> None:
    parsed = BlueprintConfigParser(leader_teleop.teleop_dual_openyam_leader).parse(
        ["--yamleaderteleopmodule.port-right=/dev/ttyACM1"], environ={}
    )
    config = YamLeaderTeleopModuleConfig(**parsed.module_kwargs(YamLeaderTeleopModule.name))
    assert config.sides() == ("right",)
    assert config.port_right == "/dev/ttyACM1"


def test_operator_blueprint_joins_the_robot_bus_as_a_client() -> None:
    blueprint = leader_teleop.teleop_dual_openyam_leader_operator
    assert _types(blueprint) == [YamLeaderTeleopModule]
    assert blueprint.global_config_overrides["zenoh_mode"] == "client"
    assert blueprint.global_config_overrides["serve_coordinator_rpc"] is False


def test_follower_blueprint_dials_the_router_and_takes_can_ports() -> None:
    blueprint = leader_teleop.teleop_dual_openyam_leader_follower
    assert _types(blueprint) == [DualOpenYamCoordinator, ManipulationModule]
    assert blueprint.global_config_overrides["zenoh_connect"] == leader_teleop.DUAL_OPENYAM_ROUTER
    assert leader_teleop.DUAL_OPENYAM_ROUTER == "tcp/127.0.0.1:7447"
    assert _kwargs(blueprint, ManipulationModule)["visualization"].host == "0.0.0.0"
    parsed = BlueprintConfigParser(blueprint).parse(
        [
            "--controlcoordinator.left-can-port=follower_l",
            "--controlcoordinator.right-can-port=follower_r",
        ],
        environ={},
    )
    kwargs = parsed.module_kwargs("ControlCoordinator")
    assert (kwargs["left_can_port"], kwargs["right_can_port"]) == ("follower_l", "follower_r")
