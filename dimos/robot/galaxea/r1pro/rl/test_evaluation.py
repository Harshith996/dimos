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

from dimos.robot.galaxea.r1pro.rl.evaluation import (
    EpisodeTrace,
    evaluate_episode,
    orientation_error_rad,
)


def trace_fixture() -> EpisodeTrace:
    times = np.arange(100, dtype=float) * 0.01
    target = np.zeros((100, 2, 3))
    target[:, :, 0] = times[:, None] ** 2
    quat = np.tile([1.0, 0, 0, 0], (100, 2, 1))
    return EpisodeTrace(
        times,
        target,
        target.copy(),
        quat,
        quat.copy(),
        times.copy(),
        np.ones(100, bool),
        np.zeros(100, bool),
        np.zeros(100, bool),
    )


def test_literal_pose_errors_and_quaternion_sign() -> None:
    trace = trace_fixture()
    actual = trace.actual_positions_m + np.array([0.03, 0.04, 0])
    quat = np.tile([0.0, 0, 0, 1], (100, 2, 1))
    result = evaluate_episode(
        replace(trace, actual_positions_m=actual, actual_quaternions_wxyz=quat)
    )
    assert result["position_rmse_m"] == pytest.approx(0.05)
    assert result["orientation_rmse_rad"] == pytest.approx(np.pi)
    np.testing.assert_array_equal(
        orientation_error_rad(trace.target_quaternions_wxyz, -trace.target_quaternions_wxyz),
        np.zeros((100, 2)),
    )


def test_lag_is_bounded_and_does_not_hide_unaligned_error() -> None:
    trace = trace_fixture()
    actual = np.concatenate(
        [trace.actual_positions_m[:1].repeat(3, axis=0), trace.actual_positions_m[:-3]]
    )
    result = evaluate_episode(replace(trace, actual_positions_m=actual), max_lag_s=0.05)
    assert result["lag_s"] == pytest.approx(0.03)
    assert result["aligned_position_rmse_m"] == pytest.approx(0)
    assert result["position_rmse_m"] > 0.01
    assert result["lag_samples"] == 90


def test_faults_counted_without_fabricated_tracking_success() -> None:
    trace = trace_fixture()
    stamps = trace.target_stamps_s.copy()
    valid = trace.target_valid.copy()
    valid[:10] = False
    stamps[10:20] -= 1
    stamps[20:25] += 1
    collision = trace.collision.copy()
    collision[30:32] = True
    limits = trace.limit_violation.copy()
    limits[40] = True
    result = evaluate_episode(
        replace(
            trace,
            target_stamps_s=stamps,
            target_valid=valid,
            collision=collision,
            limit_violation=limits,
        )
    )
    assert result["dropout_samples"] == 10
    assert result["timeout_samples"] == 10
    assert result["future_stamp_samples"] == 5
    assert result["tracking_samples"] == 75
    assert result["collision_samples"] == 2
    assert result["limit_violation_samples"] == 1
    absent = evaluate_episode(replace(trace, target_valid=np.zeros(100, bool)))
    assert absent["position_rmse_m"] is None
    assert absent["lag_s"] is None


def test_smoothness_and_bad_trace() -> None:
    trace = trace_fixture()
    result = evaluate_episode(trace)
    assert result["wrist_acceleration_rms_m_s2"] == pytest.approx(2)
    assert result["wrist_jerk_rms_m_s3"] == pytest.approx(0, abs=1e-8)
    times = trace.times_s.copy()
    times[2] = times[1]
    with pytest.raises(ValueError, match="increasing"):
        evaluate_episode(replace(trace, times_s=times))


def test_missing_orientation_is_masked_and_early_failure_stays_failure() -> None:
    trace = trace_fixture()
    mask = np.zeros((100, 2), dtype=bool)
    result = evaluate_episode(
        replace(trace, orientation_valid=mask), requested_duration_s=2, terminated_early=True
    )
    assert result["orientation_rmse_rad"] is None
    assert result["left_orientation_p95_rad"] is None
    assert result["position_rmse_m"] == 0
    assert result["completion_fraction"] == pytest.approx(0.495)
    assert result["completed"] == 0
    left_only = np.ones((100, 2), dtype=bool)
    left_only[:, 1] = False
    result = evaluate_episode(replace(trace, position_valid=left_only))
    assert result["right_position_p95_m"] is None
    assert result["left_position_p95_m"] == 0
    assert result["lag_s"] is None


def test_irregular_terminal_substep_retains_failed_completion() -> None:
    times = np.array([0.0, 0.02, 0.04, 0.042])
    positions = np.zeros((4, 2, 3))
    positions[:, :, 0] = times[:, None] ** 2
    quaternions = np.tile([1.0, 0, 0, 0], (4, 2, 1))
    trace = EpisodeTrace(
        times,
        positions,
        positions,
        quaternions,
        quaternions,
        times,
        np.ones(4, bool),
        np.array([False, False, False, True]),
        np.zeros(4, bool),
    )
    metrics = evaluate_episode(trace, requested_duration_s=0.4, terminated_early=True)
    assert metrics["completed"] == 0
    assert metrics["completion_fraction"] == pytest.approx(0.105)
    assert metrics["observed_duration_s"] == 0.042
    assert metrics["collision_samples"] == 1
    assert metrics["lag_s"] is None
    assert metrics["lag_uniform_timestamps"] == 0
    assert metrics["wrist_acceleration_rms_m_s2"] == pytest.approx(2)
    assert metrics["wrist_jerk_rms_m_s3"] == pytest.approx(0, abs=1e-8)
