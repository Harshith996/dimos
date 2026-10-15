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


"""Small same-model differential IK baseline for isolated simulation comparison."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.manipulation.planning.utils.kinematics_utils import (
    compute_error_twist,
    damped_pseudoinverse,
)
from dimos.robot.galaxea.r1pro.rl.env import TrackingEnv
from dimos.robot.galaxea.r1pro.rl.model import ARM_JOINT_NAMES, WRIST_SITE_NAMES


class DifferentialIKController:
    """Map current world-frame pose errors into the shared normalized arm rates.

    This reuses DimOS differential IK math, with actual MuJoCo site Jacobians.
    It is not the production Pink controller and has no hardware interface.
    The environment retains all command limits and safety termination behavior.
    """

    def __init__(self, env: TrackingEnv, gain: float = 2.0, damping: float = 0.05) -> None:
        if not np.isfinite([gain, damping]).all() or gain <= 0 or damping <= 0:
            raise ValueError("IK gain and damping must be finite and positive.")
        self._env = env
        self._gain = gain
        self._damping = damping
        self._site_ids = [env.model.site(name).id for name in WRIST_SITE_NAMES]
        joint_ids = np.array([env.model.joint(name).id for name in ARM_JOINT_NAMES])
        self._velocity_ids = env.model.jnt_dofadr[joint_ids]

    def __call__(self, observation: NDArray[np.float64]) -> NDArray[np.float64]:
        """Read only current state and target; computing an action never steps physics."""
        del observation
        target = self._env.current_target
        if (
            target is None
            or not target.valid
            or self._env.time_s - target.stamp_s > self._env.config.max_target_age_s
        ):
            return np.zeros(14)
        jacobians = []
        twists = []
        for wrist, site_id in enumerate(self._site_ids):
            rows = np.repeat([target.position_valid[wrist], target.orientation_valid[wrist]], 3)
            if not rows.any():
                continue
            current = np.eye(4)
            current[:3, :3] = self._env.data.site_xmat[site_id].reshape(3, 3)
            current[:3, 3] = self._env.data.site_xpos[site_id]
            desired = np.eye(4)
            desired[:3, 3] = target.positions_m[wrist]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, target.quaternions_wxyz[wrist])  # type: ignore[attr-defined]  # Compiled MuJoCo API lacks this stub.
            desired[:3, :3] = rotation.reshape(3, 3)
            twist = compute_error_twist(current, desired, self._gain)
            jacobian_position = np.zeros((3, self._env.model.nv))
            jacobian_rotation = np.zeros((3, self._env.model.nv))
            mujoco.mj_jacSite(  # type: ignore[attr-defined]  # Compiled MuJoCo API lacks this stub.
                self._env.model, self._env.data, jacobian_position, jacobian_rotation, site_id
            )
            jacobian = np.vstack((jacobian_position, jacobian_rotation))[:, self._velocity_ids]
            jacobians.append(jacobian[rows])
            twists.append(twist[rows])
        if not jacobians:
            return np.zeros(14)
        velocity = damped_pseudoinverse(np.vstack(jacobians), self._damping) @ np.concatenate(
            twists
        )
        return np.asarray(
            np.clip(velocity / self._env.config.max_joint_target_rate_rad_s, -1.0, 1.0),
            dtype=np.float64,
        )
