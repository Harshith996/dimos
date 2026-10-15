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


"""Actual-model checks for the differential IK comparison controller."""

from dataclasses import replace

import mujoco
import numpy as np
from numpy.typing import NDArray
import pytest

from dimos.robot.galaxea.r1pro.rl.controllers import DifferentialIKController
from dimos.robot.galaxea.r1pro.rl.env import TrackingEnv
from dimos.robot.galaxea.r1pro.rl.model import ARM_JOINT_NAMES, WRIST_SITE_NAMES, load_model
from dimos.robot.galaxea.r1pro.rl.targets import TargetFrame


@pytest.fixture
def env() -> TrackingEnv:
    result = TrackingEnv(load_model(), ARM_JOINT_NAMES, WRIST_SITE_NAMES)
    result.reset()
    return result


def pose_at(env: TrackingEnv, qpos: NDArray[np.float64]) -> TargetFrame:
    data = mujoco.MjData(env.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(env.model, data)
    sites = [env.model.site(name).id for name in WRIST_SITE_NAMES]
    quaternions = np.empty((2, 4))
    for i, site_id in enumerate(sites):
        mujoco.mju_mat2Quat(quaternions[i], data.site_xmat[site_id])
    return TargetFrame(np.asarray(data.site_xpos[sites], dtype=np.float64), quaternions, 0.0)


def orientation_error(current: TargetFrame, target: TargetFrame) -> float:
    dot = np.sum(current.quaternions_wxyz * target.quaternions_wxyz, axis=1)
    return float(np.sum(2 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))))


def test_zero_error_produces_zero_action_without_mutating_physics(env: TrackingEnv) -> None:
    controller = DifferentialIKController(env)
    before_qpos, before_qvel = env.data.qpos.copy(), env.data.qvel.copy()
    action = controller(env.observation())
    np.testing.assert_array_equal(action, np.zeros(14))
    np.testing.assert_array_equal(env.data.qpos, before_qpos)
    np.testing.assert_array_equal(env.data.qvel, before_qvel)
    assert env.time_s == 0.0


def test_position_step_reduces_current_kinematic_error(env: TrackingEnv) -> None:
    qtarget = np.zeros(14)
    qtarget[1] = 0.02
    target = replace(pose_at(env, qtarget), orientation_valid=np.zeros(2, dtype=bool))
    env.set_target(target)
    action = DifferentialIKController(env)(env.observation())
    initial = pose_at(env, np.zeros(14))
    moved = pose_at(env, action * 0.25 * 0.02)
    initial_error = np.linalg.norm(initial.positions_m - target.positions_m)
    moved_error = np.linalg.norm(moved.positions_m - target.positions_m)
    assert moved_error < initial_error
    np.testing.assert_allclose(action[7:], np.zeros(7), atol=1e-12)


def test_orientation_step_reduces_current_kinematic_error(env: TrackingEnv) -> None:
    qtarget = np.zeros(14)
    qtarget[6] = 0.02
    target = replace(pose_at(env, qtarget), position_valid=np.zeros(2, dtype=bool))
    env.set_target(target)
    action = DifferentialIKController(env)(env.observation())
    initial = pose_at(env, np.zeros(14))
    moved = pose_at(env, action * 0.25 * 0.02)
    assert orientation_error(moved, target) < orientation_error(initial, target)


def test_missing_orientation_does_not_create_rotation_command(env: TrackingEnv) -> None:
    qtarget = np.zeros(14)
    qtarget[6] = 0.2
    orientation = pose_at(env, qtarget).quaternions_wxyz
    target = replace(
        env.current_pose_target(),
        quaternions_wxyz=orientation,
        orientation_valid=np.zeros(2, dtype=bool),
    )
    env.set_target(target)
    np.testing.assert_array_equal(DifferentialIKController(env)(env.observation()), np.zeros(14))


def test_singular_pose_is_finite_bounded_deterministic_and_uses_shared_adapter(
    env: TrackingEnv,
) -> None:
    target = replace(
        env.current_pose_target(),
        positions_m=np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]),
    )
    env.set_target(target)
    controller = DifferentialIKController(env)
    first = controller(env.observation())
    repeated = controller(env.observation())
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) == 1.0
    np.testing.assert_array_equal(first, repeated)
    _, _, _, _, info = env.step(first)
    np.testing.assert_allclose(env.command_positions_rad, first * 0.005, atol=1e-12)
    assert info["raw_action"] == first.tolist()
    limits = env.model.jnt_range
    assert np.all(env.command_positions_rad >= limits[:, 0] + env.config.joint_margin_rad)
    assert np.all(env.command_positions_rad <= limits[:, 1] - env.config.joint_margin_rad)


def test_invalid_target_holds_command_and_environment_terminates(env: TrackingEnv) -> None:
    env.set_target(replace(env.current_pose_target(), valid=False))
    action = DifferentialIKController(env)(env.observation())
    np.testing.assert_array_equal(action, np.zeros(14))
    _, _, terminated, _, info = env.step(action)
    assert terminated is True
    assert info["termination_reason"] == "target_dropout"


def test_controller_target_obeys_environment_joint_margin(env: TrackingEnv) -> None:
    limits = env.model.jnt_range
    initial = np.zeros(14)
    initial[6] = limits[6, 1] - 0.011
    env.reset(qpos=initial)
    beyond_limit = initial.copy()
    beyond_limit[6] += 0.2
    env.set_target(replace(pose_at(env, beyond_limit), position_valid=np.zeros(2, dtype=bool)))
    action = DifferentialIKController(env)(env.observation())
    assert action[6] > 0.2
    _, _, _, _, info = env.step(action)
    assert env.command_positions_rad[6] == pytest.approx(limits[6, 1] - 0.01)
    assert info["action_clipped"] is True
