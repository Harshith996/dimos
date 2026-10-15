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


"""Independent offline metrics. Alignment never replaces unaligned tracking error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def orientation_error_rad(actual: Array, target: Array) -> Array:
    """Shortest quaternion angle; q and -q represent the same orientation."""
    for quat in (actual, target):
        if not np.all(np.isfinite(quat)) or not np.allclose(
            np.linalg.norm(quat, axis=-1), 1, atol=1e-6
        ):
            raise ValueError("Orientations must be finite unit quaternions")
    return np.asarray(
        2 * np.arccos(np.clip(np.abs(np.sum(actual * target, axis=-1)), 0, 1)), dtype=np.float64
    )


@dataclass(frozen=True)
class EpisodeTrace:
    times_s: Array
    target_positions_m: Array
    actual_positions_m: Array
    target_quaternions_wxyz: Array
    actual_quaternions_wxyz: Array
    target_stamps_s: Array
    target_valid: NDArray[np.bool_]
    collision: NDArray[np.bool_]
    limit_violation: NDArray[np.bool_]
    position_valid: NDArray[np.bool_] | None = None
    orientation_valid: NDArray[np.bool_] | None = None


def evaluate_episode(
    trace: EpisodeTrace,
    *,
    target_timeout_s: float = 0.2,
    max_lag_s: float = 0.2,
    requested_duration_s: float | None = None,
    terminated_early: bool = False,
) -> dict[str, float | int | None]:
    """Report all fault samples, evaluate tracking on fresh valid references only.

    Positive lag means actual motion follows an older target. Lag search uses the
    same interior time indices for all candidates. Alignment also requires every
    reference between both indices to be fresh, so it cannot bridge dropouts.
    Smoothness uses all observed robot motion, including fault intervals.
    """
    t = np.asarray(trace.times_s, dtype=np.float64)
    n = len(t)
    expected = {
        "times_s": (n,),
        "target_positions_m": (n, 2, 3),
        "actual_positions_m": (n, 2, 3),
        "target_quaternions_wxyz": (n, 2, 4),
        "actual_quaternions_wxyz": (n, 2, 4),
        "target_stamps_s": (n,),
        "target_valid": (n,),
        "collision": (n,),
        "limit_violation": (n,),
    }
    if (
        n < 4
        or not np.isfinite([target_timeout_s, max_lag_s]).all()
        or target_timeout_s <= 0
        or max_lag_s < 0
    ):
        raise ValueError("Need at least four samples and valid timeout/lag bounds")
    for name, shape in expected.items():
        value = np.asarray(getattr(trace, name))
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid trace field {name}, expected finite {shape}")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Evaluation requires strictly increasing timestamps")
    uniform_timestamps = bool(np.allclose(dt, dt[0], rtol=1e-5, atol=1e-9))
    if requested_duration_s is not None and (
        not np.isfinite(requested_duration_s) or requested_duration_s <= 0
    ):
        raise ValueError("Requested duration must be positive")
    masks = []
    for mask in (trace.position_valid, trace.orientation_valid):
        value = np.ones((n, 2), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if value.shape != (n, 2):
            raise ValueError("Pose masks must have shape (samples, 2)")
        masks.append(value)
    orientation = orientation_error_rad(
        trace.actual_quaternions_wxyz, trace.target_quaternions_wxyz
    )
    age = t - trace.target_stamps_s
    valid = np.asarray(trace.target_valid, dtype=bool)
    stale = valid & (age > target_timeout_s)
    future = valid & (age < 0)
    fresh = valid & ~stale & ~future
    position = np.linalg.norm(trace.actual_positions_m - trace.target_positions_m, axis=-1)
    position_mask = fresh[:, None] & masks[0]
    orientation_mask = fresh[:, None] & masks[1]
    speed = np.diff(trace.actual_positions_m, axis=0) / dt[:, None, None]
    speed_times = (t[1:] + t[:-1]) / 2
    acceleration = np.diff(speed, axis=0) / np.diff(speed_times)[:, None, None]
    acceleration_times = (speed_times[1:] + speed_times[:-1]) / 2
    jerk = np.diff(acceleration, axis=0) / np.diff(acceleration_times)[:, None, None]
    result: dict[str, float | int | None] = {
        "samples": n,
        "tracking_samples": int(np.count_nonzero(fresh)),
        "dropout_samples": int(np.count_nonzero(~valid)),
        "timeout_samples": int(np.count_nonzero(stale)),
        "future_stamp_samples": int(np.count_nonzero(future)),
        "collision_samples": int(np.count_nonzero(trace.collision)),
        "limit_violation_samples": int(np.count_nonzero(trace.limit_violation)),
        "position_rmse_m": float(np.sqrt(np.mean(position[position_mask] ** 2)))
        if np.any(position_mask)
        else None,
        "position_p95_m": float(np.percentile(position[position_mask], 95))
        if np.any(position_mask)
        else None,
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation[orientation_mask] ** 2)))
        if np.any(orientation_mask)
        else None,
        "wrist_speed_rms_m_s": float(np.sqrt(np.mean(np.sum(speed**2, axis=-1)))),
        "wrist_acceleration_rms_m_s2": float(np.sqrt(np.mean(np.sum(acceleration**2, axis=-1)))),
        "wrist_jerk_rms_m_s3": float(np.sqrt(np.mean(np.sum(jerk**2, axis=-1)))),
        "lag_s": None,
        "aligned_position_rmse_m": None,
        "lag_samples": 0,
        "lag_bound_s": float(max_lag_s),
        "lag_uniform_timestamps": int(uniform_timestamps),
        "observed_duration_s": float(t[-1] - t[0]),
        "requested_duration_s": requested_duration_s,
        "completion_fraction": min(1.0, float((t[-1] - t[0]) / requested_duration_s))
        if requested_duration_s is not None
        else None,
        "completed": int(
            not terminated_early
            and (requested_duration_s is None or t[-1] - t[0] >= requested_duration_s - 1e-9)
        ),
    }
    for side, label in enumerate(("left", "right")):
        for name, values, mask in (
            ("position", position, position_mask),
            ("orientation", orientation, orientation_mask),
        ):
            selected = values[mask[:, side], side]
            unit = "m" if name == "position" else "rad"
            for percentile in (50, 95, 100):
                key = f"{label}_{name}_p{percentile}_{unit}"
                result[key] = float(np.percentile(selected, percentile)) if len(selected) else None
    if not uniform_timestamps:
        return result
    bound = int(np.floor(max_lag_s / dt[0] + 1e-8))
    if bound * 2 >= n:
        return result
    indices = np.arange(bound, n - bound)
    # Keep the population identical for every offset and reject gaps in the window.
    indices = indices[
        np.array(
            [
                (fresh[i - bound : i + bound + 1, None] & masks[0][i - bound : i + bound + 1]).all()
                for i in indices
            ]
        )
    ]
    if len(indices) < 4:
        return result
    candidates: list[tuple[float, int]] = []
    for offset in range(-bound, bound + 1):
        delta = trace.actual_positions_m[indices] - trace.target_positions_m[indices - offset]
        error = float(np.sqrt(np.mean(np.sum(delta**2, axis=-1))))
        candidates.append((error, offset))
    error, offset = min(candidates, key=lambda pair: (round(pair[0], 12), abs(pair[1])))
    result.update(
        lag_s=float(offset * dt[0]), aligned_position_rmse_m=error, lag_samples=len(indices)
    )
    return result
