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


"""CPU MuJoCo tracking environment with a simulation-only arm position adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.rl.model import collision_pairs
from dimos.robot.galaxea.r1pro.rl.reward import (
    RewardInput,
    smoke_tracking_reward,
    tracking_reward_terms,
)
from dimos.robot.galaxea.r1pro.rl.targets import TargetFrame


@dataclass(frozen=True)
class TrackingConfig:
    control_dt_s: float = 0.02
    episode_steps: int = 100
    max_target_age_s: float = 0.1
    max_joint_target_rate_rad_s: float = 0.25
    joint_margin_rad: float = 0.01
    terminate_on_collision: bool = True
    collision_penetration_tolerance_m: float = 1e-5

    def __post_init__(self) -> None:
        numbers = (
            self.control_dt_s,
            self.max_target_age_s,
            self.max_joint_target_rate_rad_s,
            self.joint_margin_rad,
            self.collision_penetration_tolerance_m,
        )
        if not all(np.isfinite(x) and x > 0 for x in numbers) or self.episode_steps < 1:
            raise ValueError("Tracking configuration must have finite positive bounds.")


class TrackingEnv:
    """The adapter expresses bounded arm joint target rates, never hardware commands.

    Reset returns the initial observation. Step returns the terminal observation and
    requires an explicit reset after termination or truncation. A missing/stale
    target holds the previous position command and terminates without advancing
    physics. Safety events are reported separately from the replaceable reward.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: Sequence[str],
        wrist_sites: Sequence[str],
        config: TrackingConfig = TrackingConfig(),
        reward_fn: Callable[[RewardInput], float] = smoke_tracking_reward,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config
        self._reward_fn = reward_fn
        self._joint_ids = np.array([model.joint(name).id for name in joint_names])
        self._qids = model.jnt_qposadr[self._joint_ids]
        self._vids = model.jnt_dofadr[self._joint_ids]
        self._actuator_ids = np.array(
            [model.actuator(name + "_position").id for name in joint_names]
        )
        self._site_ids = np.array([model.site(name).id for name in wrist_sites])
        if len(self._joint_ids) != 14 or len(self._site_ids) != 2 or model.nq != 14:
            raise ValueError("Initial curriculum requires a welded base/torso and 14 arm joints.")
        if not np.all(model.jnt_limited[self._joint_ids]):
            raise ValueError("Every controlled joint must have finite model limits.")
        self._limits = model.jnt_range[self._joint_ids].copy()
        self._substeps = round(config.control_dt_s / model.opt.timestep)
        if self._substeps < 1 or not np.isclose(
            self._substeps * model.opt.timestep, config.control_dt_s
        ):
            raise ValueError("Control period must be an integer multiple of physics timestep.")
        if np.any(self._limits[:, 1] - self._limits[:, 0] <= 2 * config.joint_margin_rad):
            raise ValueError("Joint margin leaves no valid range.")
        self.command_positions_rad = np.zeros(14)
        self._target: TargetFrame | None = None
        self._steps = 0
        self._done = True
        self._rejected_stale_input = False

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    @property
    def episode_length(self) -> int:
        return self._steps

    def joint_positions(self) -> NDArray[np.float64]:
        return np.asarray(self.data.qpos[self._qids], dtype=np.float64).copy()

    def joint_velocities(self) -> NDArray[np.float64]:
        return np.asarray(self.data.qvel[self._vids], dtype=np.float64).copy()

    def wrist_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        quat = np.empty((2, 4))
        for i, site_id in enumerate(self._site_ids):
            mujoco.mju_mat2Quat(quat[i], self.data.site_xmat[site_id])  # type: ignore[attr-defined]  # Compiled MuJoCo API lacks this stub.
        return self.data.site_xpos[self._site_ids].copy(), quat

    def current_pose_target(self) -> TargetFrame:
        positions, quaternions = self.wrist_pose()
        return TargetFrame(positions_m=positions, quaternions_wxyz=quaternions, stamp_s=self.time_s)

    def reset(
        self, *, seed: int = 0, qpos: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Reset deterministically; seed is reserved for replaceable target sources."""
        del seed
        initial = np.zeros(14) if qpos is None else np.asarray(qpos, dtype=float)
        if initial.shape != (14,) or not np.all(np.isfinite(initial)):
            raise ValueError("Reset requires 14 finite joint positions.")
        if np.any(initial < self._limits[:, 0]) or np.any(initial > self._limits[:, 1]):
            raise ValueError("Reset positions violate model joint limits.")
        mujoco.mj_resetData(self.model, self.data)  # type: ignore[attr-defined]  # Compiled MuJoCo API lacks this stub.
        self.data.qpos[self._qids] = initial
        self.command_positions_rad = initial.copy()
        self.data.ctrl[self._actuator_ids] = initial
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._done = False
        self._rejected_stale_input = False
        self._target = self.current_pose_target()
        return self.observation()

    def observation(self) -> NDArray[np.float64]:
        """Use measured state, current target, previous command and current target age only."""
        positions, quaternions = self.wrist_pose()
        target = self._target
        if target is None:
            raise RuntimeError("Reset must provide the initial current target.")
        return np.concatenate(
            (
                self.joint_positions(),
                self.joint_velocities(),
                positions.ravel(),
                quaternions.ravel(),
                np.where(target.position_valid[:, None], target.positions_m, 0.0).ravel(),
                np.where(target.orientation_valid[:, None], target.quaternions_wxyz, 0.0).ravel(),
                target.position_valid.astype(float),
                target.orientation_valid.astype(float),
                self.command_positions_rad,
                [max(0.0, self.time_s - target.stamp_s)],
            )
        )

    def set_target(self, target: TargetFrame | None) -> None:
        """Accept only current in-order input; old input never refreshes the watchdog."""
        self._rejected_stale_input = False
        if target is None:
            return
        if target.stamp_s > self.time_s + 1e-9:
            raise ValueError("Deployable observations cannot contain a future target.")
        if self._target is not None and target.stamp_s < self._target.stamp_s:
            self._rejected_stale_input = True
            return
        self._target = target

    def step(
        self, action: NDArray[np.float64], target: TargetFrame | None = None
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode ended; reset before the next action.")
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain 14 finite normalized arm rates.")
        if target is not None:
            self.set_target(target)
        assert self._target is not None
        target_age_s = self.time_s - self._target.stamp_s
        if not self._target.valid or target_age_s > self.config.max_target_age_s + 1e-9:
            self._done = True
            return (
                self.observation(),
                0.0,
                True,
                False,
                {
                    "termination_reason": "target_dropout"
                    if not self._target.valid
                    else "target_timeout",
                    "target_age_s": target_age_s,
                    "collision_count": len(
                        collision_pairs(
                            self.model, self.data, self.config.collision_penetration_tolerance_m
                        )
                    ),
                    "raw_contact_count": int(self.data.ncon),
                    "rejected_stale_input": self._rejected_stale_input,
                    "joint_limit_violation": False,
                },
            )
        previous = self.command_positions_rad.copy()
        requested = previous + np.clip(action, -1, 1) * (
            self.config.control_dt_s * self.config.max_joint_target_rate_rad_s
        )
        self.command_positions_rad = np.clip(
            requested,
            self._limits[:, 0] + self.config.joint_margin_rad,
            self._limits[:, 1] - self.config.joint_margin_rad,
        )
        self.data.ctrl[self._actuator_ids] = self.command_positions_rad
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            qpos = self.joint_positions()
            if (
                not np.isfinite(qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or np.any(qpos < self._limits[:, 0])
                or np.any(qpos > self._limits[:, 1])
                or (
                    self.config.terminate_on_collision
                    and collision_pairs(
                        self.model, self.data, self.config.collision_penetration_tolerance_m
                    )
                )
            ):
                break
        self._steps += 1
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        qpos = self.joint_positions()
        limit_violation = bool(
            np.any(qpos < self._limits[:, 0]) or np.any(qpos > self._limits[:, 1])
        )
        forbidden_pairs = collision_pairs(
            self.model, self.data, self.config.collision_penetration_tolerance_m
        )
        collisions = len(forbidden_pairs)
        terminated = (
            not finite or limit_violation or (self.config.terminate_on_collision and collisions > 0)
        )
        truncated = self._steps >= self.config.episode_steps and not terminated
        reason = (
            "nonfinite"
            if not finite
            else "joint_limit"
            if limit_violation
            else "collision"
            if terminated
            else "time_limit"
            if truncated
            else "running"
        )
        self._done = terminated or truncated
        positions, quaternions = self.wrist_pose()
        reward_input = RewardInput(
            positions,
            self._target.positions_m,
            quaternions,
            self._target.quaternions_wxyz,
            self.command_positions_rad - previous,
            self._target.position_valid,
            self._target.orientation_valid,
            self.config.control_dt_s,
            self.joint_velocities(),
        )
        reward = self._reward_fn(reward_input) if finite else 0.0
        if not np.isfinite(reward):
            raise ValueError("Reward function produced a nonfinite value.")
        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            {
                "termination_reason": reason,
                "target_age_s": target_age_s,
                "collision_count": collisions,
                "joint_limit_violation": limit_violation,
                "reward_terms": tracking_reward_terms(reward_input),
                "rejected_stale_input": self._rejected_stale_input,
                "raw_contact_count": int(self.data.ncon),
                "collision_pairs": forbidden_pairs,
                "action_clipped": bool(
                    np.any(np.abs(action) > 1) or np.any(requested != self.command_positions_rad)
                ),
                "raw_action": action.tolist(),
                "requested_positions_rad": requested.tolist(),
                "limited_positions_rad": self.command_positions_rad.tolist(),
                "position_valid": self._target.position_valid.tolist(),
                "orientation_valid": self._target.orientation_valid.tolist(),
                "raw_target_positions_m": self._target.positions_m.tolist(),
                "target_positions_m": self._target.positions_m.tolist(),
                "target_quaternions_wxyz": self._target.quaternions_wxyz.tolist(),
            },
        )
