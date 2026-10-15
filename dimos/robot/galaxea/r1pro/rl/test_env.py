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


"""Behavior checks against the actual pinned MuJoCo model and RSL adapter."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from dimos.robot.galaxea.r1pro.rl.env import TrackingConfig, TrackingEnv
from dimos.robot.galaxea.r1pro.rl.model import ARM_JOINT_NAMES, WRIST_SITE_NAMES, load_model
from dimos.robot.galaxea.r1pro.rl.reward import RewardInput, smoke_tracking_reward
from dimos.robot.galaxea.r1pro.rl.train import TrackingVecEnv, procedural_source


@pytest.fixture
def env() -> TrackingEnv:
    result = TrackingEnv(load_model(), ARM_JOINT_NAMES, WRIST_SITE_NAMES)
    result.reset()
    return result


def test_rate_bound_and_deterministic_reset(env: TrackingEnv) -> None:
    initial = env.reset(seed=42)
    first, _, terminated, _, info = env.step(np.full(14, 100.0))
    np.testing.assert_allclose(env.command_positions_rad, np.full(14, 0.005), atol=1e-12)
    assert info["action_clipped"] is True
    assert terminated is False
    np.testing.assert_array_equal(env.reset(seed=42), initial)
    repeated, _, _, _, _ = env.step(np.full(14, 100.0))
    np.testing.assert_array_equal(repeated, first)


def test_time_limit_requires_reset_and_vec_resets() -> None:
    config = TrackingConfig(episode_steps=1)
    env = TrackingEnv(load_model(), ARM_JOINT_NAMES, WRIST_SITE_NAMES, config)
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros(14))
    assert (terminated, truncated, info["termination_reason"]) == (False, True, "time_limit")
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(14))
    vector = TrackingVecEnv(config, num_envs=1)
    _, _, dones, extras = vector.step(torch.zeros((1, 14)))
    assert dones.tolist() == [True]
    assert extras["time_outs"].tolist() == [True]
    assert vector.envs[0].time_s == 0.0
    assert vector.episode_length_buf.tolist() == [0]


def test_dropout_holds_command_without_advancing_physics(env: TrackingEnv) -> None:
    target = replace(env.current_pose_target(), valid=False)
    before = env.joint_positions()
    _, reward, terminated, truncated, info = env.step(np.ones(14), target)
    assert (reward, terminated, truncated, info["termination_reason"]) == (
        0.0,
        True,
        False,
        "target_dropout",
    )
    assert env.time_s == 0.0
    np.testing.assert_array_equal(env.joint_positions(), before)
    np.testing.assert_array_equal(env.command_positions_rad, np.zeros(14))


def test_missing_and_stale_samples_expire_without_refresh(env: TrackingEnv) -> None:
    source = procedural_source(None, 42)
    source.stale_s = (0.02, 1.0)
    reasons = []
    for _ in range(7):
        _, _, terminated, _, info = env.step(np.zeros(14), source.sample(env.time_s))
        reasons.append(info["termination_reason"])
    assert reasons == [
        "running",
        "running",
        "running",
        "running",
        "running",
        "running",
        "target_timeout",
    ]
    assert terminated is True
    assert info["rejected_stale_input"] is True
    assert env.time_s == pytest.approx(0.12)


def test_future_target_rejected_and_current_target_visible_before_action(env: TrackingEnv) -> None:
    target = replace(env.current_pose_target(), stamp_s=0.02)
    with pytest.raises(ValueError, match="future"):
        env.set_target(target)
    current = replace(
        env.current_pose_target(), positions_m=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    )
    env.set_target(current)
    np.testing.assert_array_equal(env.observation()[42:48], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert env.observation().shape == (75,)


def test_masks_remove_missing_wrist_and_orientation_from_reward(env: TrackingEnv) -> None:
    positions, quaternions = env.wrist_pose()
    target = positions.copy()
    target[0] += 100.0
    state = RewardInput(
        positions,
        target,
        quaternions,
        -quaternions,
        np.zeros(14),
        np.array([False, True]),
        np.array([False, False]),
    )
    assert smoke_tracking_reward(state) == pytest.approx(4.0)
    complete = replace(
        state,
        target_positions_m=positions,
        position_valid=np.ones(2, dtype=bool),
        orientation_valid=np.ones(2, dtype=bool),
    )
    assert smoke_tracking_reward(complete) == pytest.approx(6.0)
    invalid = replace(
        complete, position_valid=np.zeros(2, dtype=bool), orientation_valid=np.zeros(2, dtype=bool)
    )
    assert smoke_tracking_reward(invalid) == 0.0


def test_current_procedural_target_reaches_actor_before_step() -> None:
    source = procedural_source(None, 42)
    vector = TrackingVecEnv(TrackingConfig(), num_envs=1, target_sources=[source.sample])
    first = vector.get_observations()["policy"][0, 42:48].numpy().copy()
    vector.step(torch.zeros((1, 14)))
    current = vector.get_observations()["policy"][0, 42:48].numpy()
    expected = source.sample(vector.envs[0].time_s)
    assert expected is not None
    np.testing.assert_allclose(current, expected.positions_m.ravel(), atol=1e-7)
    assert np.max(np.abs(current - first)) > 1e-8


def test_nonfinite_action_rejected_without_physics(env: TrackingEnv) -> None:
    with pytest.raises(ValueError, match="finite"):
        env.step(np.full(14, np.nan))
    assert env.time_s == 0.0


def test_unreachable_reference_does_not_trigger_tracking_error_termination(
    env: TrackingEnv,
) -> None:
    target = replace(env.current_pose_target(), positions_m=np.full((2, 3), 100.0))
    _, _, terminated, truncated, info = env.step(np.zeros(14), target)
    assert (terminated, truncated, info["termination_reason"]) == (False, False, "running")
    assert env.time_s == pytest.approx(0.02)
