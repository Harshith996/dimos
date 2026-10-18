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

import attrs
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


def _solver() -> DualOpenYamPinkPoseTargetSolver:
    task = _dual_openyam_webxr_tasks["left"]
    config = PoseTargetIKTaskConfig(
        joint_names=tuple(task.joint_names),
        robot_model=task.params["robot_model"],
        target_frames=_TARGET_FRAMES,
        pink=task.params["pink"],
        max_joint_velocity_rad_s=task.params["max_joint_velocity_rad_s"],
        joint_command_filter_cutoff_hz=task.params["joint_command_filter_cutoff_hz"],
    )
    return DualOpenYamPinkPoseTargetSolver(config)


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


_ELBOW_FRAME = "left_link3"
_ALL_FRAMES = (*_TARGET_FRAMES, _ELBOW_FRAME)


def _elbow_solver() -> DualOpenYamPinkPoseTargetSolver:
    config = attrs.evolve(
        _solver()._control_config, hint_frames=(_ELBOW_FRAME,), hint_position_cost=2.0
    )
    return DualOpenYamPinkPoseTargetSolver(config)


def _home_state() -> JointState:
    return JointState(name=dual_openyam_arm_joints("left"), position=DUAL_OPENYAM_HOME_PER_ARM)


@pytest.mark.self_hosted
def test_elbow_hint_is_a_position_only_task_below_the_hand() -> None:
    solver = _elbow_solver()
    state = _home_state()
    targets = solver.frame_poses(state, _ALL_FRAMES)

    assert solver.step(targets, state, 0.01) is not None
    context = next(iter(solver._control_contexts.values()))
    assert context.tasks is not None
    assert list(context.tasks[f"frame/{_ELBOW_FRAME}"].cost) == [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    assert list(context.tasks["frame/left_grasp_frame"].cost) == [8.0, 8.0, 8.0, 2.0, 2.0, 2.0]

    assert solver.step(targets, state, 0.01, inactive_frames=frozenset({_ELBOW_FRAME})) is not None
    assert list(context.tasks[f"frame/{_ELBOW_FRAME}"].cost) == [0.0] * 6


@pytest.mark.self_hosted
def test_elbow_hint_bends_the_arm_toward_the_operator_without_losing_the_hand() -> None:
    solver = _elbow_solver()
    initial = solver.frame_poses(_home_state(), _ALL_FRAMES)
    elbow = initial[_ELBOW_FRAME]
    targets = {
        **initial,
        _ELBOW_FRAME: PoseStamped(
            frame_id=elbow.frame_id,
            position=[elbow.position.x, elbow.position.y + 0.15, elbow.position.z],
            orientation=elbow.orientation,
        ),
    }

    def settle(inactive: frozenset[str]) -> dict[str, PoseStamped]:
        solver.reset()
        state = _home_state()
        for _ in range(50):
            command = solver.step(targets, state, 0.01, inactive_frames=inactive)
            assert command is not None
            state = command
        return solver.frame_poses(state, _ALL_FRAMES)

    steered = settle(frozenset())
    ignored = settle(frozenset({_ELBOW_FRAME}))

    hand_error = np.linalg.norm(
        steered["left_grasp_frame"].position.to_numpy()
        - initial["left_grasp_frame"].position.to_numpy()
    )
    assert steered[_ELBOW_FRAME].position.y - elbow.position.y > 0.001
    assert hand_error < 0.01
    assert abs(ignored[_ELBOW_FRAME].position.y - elbow.position.y) < 0.0005
