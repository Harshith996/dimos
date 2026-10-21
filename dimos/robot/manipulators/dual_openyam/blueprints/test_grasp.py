# Copyright 2025-2026 Dimensional Inc.
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

import math

import pytest

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, BlueprintAtom
from dimos.core.module import ModuleBase
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.memory.utils.validation import validate_identifier
from dimos.robot.manipulators.dual_openyam.blueprints.agentic import (
    dual_openyam_agent_prompt,
    dual_openyam_grasp_agent,
)
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.blueprints.grasp import (
    DUAL_OPENYAM_GRIPPER_TASKS,
    DUAL_OPENYAM_RECORD_TOPICS,
    DUAL_OPENYAM_VIEW_TOPICS,
    dual_openyam_grasp,
    dual_openyam_grasp_model_config,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule


def _atom(blueprint: Blueprint, module: type[ModuleBase]) -> BlueprintAtom:
    return next(atom for atom in blueprint.active_blueprints if atom.module is module)


def _modules(blueprint: Blueprint) -> set[type[ModuleBase]]:
    return {atom.module for atom in blueprint.active_blueprints}


def test_model_keeps_both_arms_with_a_tcp_each() -> None:
    config = dual_openyam_grasp_model_config()
    assert config.base_pose.frame_id == "world"
    assert {group.name: group.tip_link for group in config.planning_groups} == {
        "left_manipulator": "left_tcp",
        "right_manipulator": "right_tcp",
    }
    assert len(config.joint_names) == 12


def test_tcp_is_the_grasp_frame_turned_to_point_down_the_fingers() -> None:
    config = dual_openyam_grasp_model_config()
    frames = {frame.name: frame for frame in config.model._fixed_frames}
    assert frames["right_tcp"].parent == "right_grasp_frame"
    assert frames["right_tcp"].rpy == (0.0, math.pi, 0.0)
    assert frames["right_tcp"].xyz[2] == pytest.approx(-0.046)
    assert frames["left_tcp"].xyz[2] == pytest.approx(-0.046)


def test_each_planning_group_is_bound_to_its_own_gripper_task() -> None:
    tasks: list[TaskConfig] = _atom(dual_openyam_grasp, DualOpenYamCoordinator).kwargs["tasks"]
    grippers = {task.name: task.joint_names for task in tasks if task.type == "gripper"}
    assert grippers == {
        "left_arm_gripper": ["left_arm/gripper"],
        "right_arm_gripper": ["right_arm/gripper"],
    }
    bound = _atom(dual_openyam_grasp, ManipulationModule).kwargs["gripper_tasks"]
    assert bound == DUAL_OPENYAM_GRIPPER_TASKS
    assert set(bound.values()) == set(grippers)
    assert not any(
        task.type == "trajectory"
        and set(task.joint_names) & {"left_arm/gripper", "right_arm/gripper"}
        for task in tasks
    )


def test_stack_is_the_xarm_grasp_stack_with_the_workcell_rig() -> None:
    manipulation = _atom(dual_openyam_grasp, ManipulationModule).kwargs
    assert manipulation["world_frame"] == "world"
    assert "floor_z" not in manipulation
    assert manipulation["static_transforms"][0].child_frame_id == "camera_link"
    assert _atom(dual_openyam_grasp, PickAndPlaceModule).kwargs == {
        "planning_frame": "world",
        "pregrasp_offset": 0.05,
    }
    overhead = next(
        atom
        for atom in dual_openyam_grasp.active_blueprints
        if atom.module is RealSenseCamera and atom.kwargs.get("enable_pointcloud")
    )
    assert (overhead.kwargs["width"], overhead.kwargs["height"], overhead.kwargs["fps"]) == (
        640,
        480,
        30,
    )
    assert dual_openyam_grasp.global_config_overrides["record"] == "sqlite"
    bridge = _atom(dual_openyam_grasp, RerunBridgeModule).kwargs
    assert bridge["topics"] == DUAL_OPENYAM_VIEW_TOPICS
    assert bridge["rerun_open"] == "none"
    assert bridge["blueprint"]() is not None


def test_recorded_stream_names_are_valid_identifiers() -> None:
    for name in DUAL_OPENYAM_RECORD_TOPICS.split(","):
        validate_identifier(name)


def test_agent_prompt_names_both_groups_and_their_drop_poses() -> None:
    prompt = dual_openyam_agent_prompt()
    assert "left_manipulator when the object's Y is positive" in prompt
    assert "place_at 0.26 0.2 0.05 left_manipulator" in prompt
    assert "place_at 0.26 -0.2 0.05 right_manipulator" in prompt
    assert {HeuristicGraspModule, ManipulationModule} <= _modules(dual_openyam_grasp_agent)
