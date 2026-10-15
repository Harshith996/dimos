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


from dataclasses import replace

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.rl.targets import (
    CurrentTargetBuffer,
    FkTrajectorySource,
    FrozenTargetSource,
    TargetFrame,
    generate_joint_trajectory,
)


def frame(q: np.ndarray) -> TargetFrame:
    return TargetFrame(np.stack([q[:3], q[:3]]), np.array([[1.0, 0, 0, 0]] * 2), 0.0)


def test_current_target_fresh_dropout_stale_future_and_reset() -> None:
    buffer = CurrentTargetBuffer()
    original = frame(np.zeros(3))
    buffer.push(original)
    assert buffer.current(0.1, 0.2) is original
    assert buffer.current(0.21, 0.2) is None
    assert buffer.current(-0.1, 0.2) is None
    buffer.push(replace(original, stamp_s=0.2, valid=False))
    assert buffer.current(0.2, 0.2) is None
    with pytest.raises(ValueError, match="Out-of-order"):
        buffer.push(original)
    buffer.reset()
    assert buffer.current(0.2, 0.2) is None


def test_trajectories_are_deterministic_bounded_smooth_and_split_by_scenario() -> None:
    args = (np.full(3, -1.0), np.full(3, 1.0), np.zeros(3))
    train = generate_joint_trajectory(
        *args, seed=7, split="train", scenario="excursion", is_valid=lambda q: True
    )
    again = generate_joint_trajectory(
        *args, seed=7, split="train", scenario="excursion", is_valid=lambda q: True
    )
    heldout = generate_joint_trajectory(
        *args, seed=7, split="heldout", scenario="pause", is_valid=lambda q: True
    )
    assert np.array_equal(train.knots_rad, again.knots_rad)
    assert not np.array_equal(train.knots_rad[1], heldout.knots_rad[1])
    assert np.array_equal(heldout.knots_rad[1], heldout.knots_rad[2])
    values = np.stack([train.sample(t) for t in np.linspace(0, 8, 8001)])
    assert np.max(np.abs(values)) <= 0.08
    assert np.max(np.abs(np.diff(values, axis=0))) / 0.001 <= 0.2
    assert np.max(np.abs(np.diff(values, axis=0)[[0, -1]])) < 1e-8
    np.testing.assert_array_equal(train.sample(8), np.zeros(3))
    with pytest.raises(ValueError, match="Scenario"):
        generate_joint_trajectory(
            *args, seed=7, split="train", scenario="pause", is_valid=lambda q: True
        )


def test_rejects_collision_paths_and_fault_source_preserves_stale_stamp() -> None:
    args = (np.full(3, -1.0), np.full(3, 1.0), np.zeros(3))
    with pytest.raises(ValueError, match="No sampled"):
        generate_joint_trajectory(
            *args, seed=1, split="train", scenario="excursion", is_valid=lambda q: False
        )
    path = generate_joint_trajectory(
        *args, seed=1, split="heldout", scenario="reversal", is_valid=lambda q: True
    )
    source = FkTrajectorySource(path, frame, dropout_s=(1, 2), stale_s=(3, 4))
    assert source.sample(1.5) is None
    stale = source.sample(3.5)
    assert stale is not None and stale.stamp_s == 2.5
    assert source.sample(0.5).positions_m.shape == (2, 3)


def test_target_frame_owns_arrays_and_rejects_bad_orientation() -> None:
    positions = np.zeros((2, 3))
    target = TargetFrame(positions, np.array([[1.0, 0, 0, 0]] * 2), 0.0)
    positions[:] = 100
    np.testing.assert_array_equal(target.positions_m, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="unit norm"):
        TargetFrame(positions, np.zeros((2, 4)), 0.0)


def test_orientation_mask_and_manifest_faults() -> None:
    target = replace(
        frame(np.zeros(3)), orientation_valid=np.array([True, False]), provenance="user_owned"
    )
    assert not target.orientation_valid[1]
    with pytest.raises(ValueError, match="shape"):
        replace(target, position_valid=np.ones(3, bool))


def test_frozen_source_replays_exact_past_samples_and_dropout() -> None:
    source = FrozenTargetSource(
        [
            {
                "query_time_s": 0.0,
                "frame": {
                    "positions_m": np.zeros((2, 3)),
                    "quaternions_wxyz": np.array([[1.0, 0, 0, 0]] * 2),
                    "stamp_s": 0.0,
                },
            },
            {"query_time_s": 0.02, "frame": None},
            {
                "query_time_s": 0.04,
                "frame": {
                    "positions_m": np.ones((2, 3)),
                    "quaternions_wxyz": np.array([[1.0, 0, 0, 0]] * 2),
                    "stamp_s": 0.04,
                },
            },
        ]
    )
    assert source.sample(0.019).stamp_s == 0
    assert source.sample(0.021) is None
    np.testing.assert_array_equal(source.sample(0.04).positions_m, np.ones((2, 3)))
    assert source.sample(0.05) is None
    with pytest.raises(ValueError, match="future target"):
        FrozenTargetSource(
            [
                {
                    "query_time_s": 0.0,
                    "frame": {
                        "positions_m": np.zeros((2, 3)),
                        "quaternions_wxyz": np.array([[1.0, 0, 0, 0]] * 2),
                        "stamp_s": 0.01,
                    },
                },
                {"query_time_s": 0.02, "frame": None},
            ]
        )
