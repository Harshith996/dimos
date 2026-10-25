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

"""Objective tests for Dual OpenYAM WebXR teleoperation."""

import numpy as np
import pytest

from dimos.control.tasks.pose_target_ik import PoseTargetIKTaskConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
    _dual_openyam_webxr_tasks,
)
from dimos.robot.manipulators.dual_openyam.config import (
    DUAL_OPENYAM_HOME_PER_ARM,
    dual_openyam_arm_joints,
)
from dimos.robot.manipulators.dual_openyam.teleop_ik import (
    DualOpenYamPinkPoseTargetSolver,
)

_TARGET_FRAMES = ("left_grasp_frame",)


def _solver(side: str = "left") -> DualOpenYamPinkPoseTargetSolver:
    task = _dual_openyam_webxr_tasks[side]
    config = PoseTargetIKTaskConfig(
        joint_names=tuple(task.joint_names),
        robot_model=task.params["robot_model"],
        target_frames=(f"{side}_grasp_frame",),
        pink=task.params["pink"],
        max_joint_velocity_rad_s=task.params["max_joint_velocity_rad_s"],
        joint_command_filter_cutoff_hz=task.params["joint_command_filter_cutoff_hz"],
    )
    return DualOpenYamPinkPoseTargetSolver(config)


@pytest.mark.self_hosted
@pytest.mark.parametrize("side", ["left", "right"])
def test_each_arm_solver_uses_its_six_joint_home_posture(side):
    solver = _solver(side)
    seed = JointState(name=dual_openyam_arm_joints(side), position=DUAL_OPENYAM_HOME_PER_ARM)
    targets = solver.frame_poses(seed, (f"{side}_grasp_frame",))

    assert solver.step(targets, seed, 0.01) is not None
    context = next(iter(solver._control_contexts.values()))
    assert context.tasks["posture/current"].target_q == pytest.approx(DUAL_OPENYAM_HOME_PER_ARM)


@pytest.mark.self_hosted
def test_webxr_solver_matches_a1z_target_tracking_speed() -> None:
    solver = _solver()
    state = JointState(
        name=dual_openyam_arm_joints("left"),
        position=DUAL_OPENYAM_HOME_PER_ARM,
    )
    initial = solver.frame_poses(state, _TARGET_FRAMES)
    targets = {
        frame_name: PoseStamped(
            frame_id=pose.frame_id,
            position=[pose.position.x, pose.position.y, pose.position.z + 0.1],
            orientation=pose.orientation,
        )
        for frame_name, pose in initial.items()
    }

    for _ in range(25):
        command = solver.step(targets, state, 0.01)
        assert command is not None
        state = command

    current = solver.frame_poses(state, _TARGET_FRAMES)
    progress = np.mean(
        [
            current[frame_name].position.z - initial[frame_name].position.z
            for frame_name in _TARGET_FRAMES
        ]
    )
    assert progress >= 0.09
