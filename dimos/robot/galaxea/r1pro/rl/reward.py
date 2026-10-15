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


"""Replaceable smoke reward, with no claim of task or hardware validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RewardInput:
    positions_m: NDArray[np.float64]
    target_positions_m: NDArray[np.float64]
    quaternions_wxyz: NDArray[np.float64]
    target_quaternions_wxyz: NDArray[np.float64]
    action_delta_rad: NDArray[np.float64]
    position_valid: NDArray[np.bool_]
    orientation_valid: NDArray[np.bool_]
    control_dt_s: float = 0.02
    joint_velocities_rad_s: NDArray[np.float64] | None = None


def tracking_reward_terms(state: RewardInput) -> dict[str, float]:
    """Normalize errors by explicit provisional physical scales and valid wrists."""
    distance_sq = np.sum((state.positions_m - state.target_positions_m) ** 2, axis=1)
    dot = np.sum(state.quaternions_wxyz * state.target_quaternions_wxyz, axis=1)
    angle = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    position = 0.5 * (np.exp(-distance_sq / 0.20**2) + np.exp(-distance_sq / 0.05**2))
    orientation = 0.5 * (np.exp(-(angle**2) / 0.70**2) + np.exp(-(angle**2) / 0.20**2))
    return {
        "command_rate_rms_rad_s": float(
            np.sqrt(np.mean((state.action_delta_rad / state.control_dt_s) ** 2))
        ),
        "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(state.joint_velocities_rad_s**2)))
        if state.joint_velocities_rad_s is not None
        else 0.0,
        "position": float(np.mean(position[state.position_valid]))
        if state.position_valid.any()
        else 0.0,
        "orientation": float(np.mean(orientation[state.orientation_valid]))
        if state.orientation_valid.any()
        else 0.0,
    }


def smoke_tracking_reward(state: RewardInput) -> float:
    """Give a provisional dense tracking signal; constraints live in the environment."""
    # Rate and velocity costs are logged with zero weights until reward calibration.
    terms = tracking_reward_terms(state)
    return 4.0 * terms["position"] + 2.0 * terms["orientation"]
