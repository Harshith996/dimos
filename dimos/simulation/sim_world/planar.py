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

"""A robot on a planar holonomic base: x, y and yaw driven at the commanded velocity."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.msgs.sim_msgs.Contacts import Part

CONTROL_DT = 0.02
VELOCITY_TIME_CONSTANT = 0.05
PLANAR_JOINTS = (
    ("x", mujoco.mjtJoint.mjJNT_SLIDE, (1.0, 0.0, 0.0)),
    ("y", mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 1.0, 0.0)),
    ("yaw", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0)),
)


def add_planar_joints(spec: mujoco.MjSpec, base_body: str) -> None:
    """Give the base body x, y and yaw joints, each with an untuned velocity actuator."""
    base = spec.body(base_body)
    for name, kind, axis in PLANAR_JOINTS:
        joint = base.add_joint()
        joint.name = name
        joint.type = kind
        joint.axis = axis
        actuator = spec.add_actuator()
        actuator.name = name
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_velocity(kv=0.0)


def tune_velocity_gains(model: mujoco.MjModel) -> None:
    """Gains that reach the commanded velocity in VELOCITY_TIME_CONSTANT, from the compiled inertia."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, _, _ in PLANAR_JOINTS:
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        kv = data.qM[model.dof_Madr[dof]] / VELOCITY_TIME_CONSTANT
        model.actuator_gainprm[actuator, 0] = kv
        model.actuator_biasprm[actuator, 2] = -kv


class PlanarBase:
    """A planar-jointed robot with geoms named part_n, tracking a body-frame velocity command with no slip."""

    dt = CONTROL_DT
    joint_names: tuple[str, ...] = ()

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, base_body: str, parts: Iterable[Part]
    ) -> None:
        self.model = model
        self.data = data
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name, _, _ in PLANAR_JOINTS
        ]
        self._qpos_adr = np.array([model.jnt_qposadr[j] for j in joints])
        self._dof_adr = np.array([model.jnt_dofadr[j] for j in joints])
        self._actuators = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name, _, _ in PLANAR_JOINTS
            ]
        )
        known = set(parts)
        self._parts: dict[int, Part] = {}
        for geom in range(model.ngeom):
            part = model.geom(geom).name.rsplit("_", 1)[0]
            if part in known:
                self._parts[geom] = part

    def reset(self, x: float, y: float, z_feet: float, yaw: float) -> None:
        """Set the base body on the floor at z_feet, at rest."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)  # type: ignore[attr-defined]
        m.body_pos[self._body][2] = z_feet
        d.qpos[self._qpos_adr] = [x, y, yaw]
        mujoco.mj_forward(m, d)

    def tick(
        self, command: NDArray[np.float64], on_substep: Callable[[], None] | None = None
    ) -> None:
        """Advance one control tick tracking a body-frame (vx, vy, wz) command."""
        vx, vy, wz = command
        for _ in range(self.substeps):
            yaw = self.data.qpos[self._qpos_adr[2]]
            c, s = math.cos(yaw), math.sin(yaw)
            self.data.ctrl[self._actuators] = [c * vx - s * vy, s * vx + c * vy, wz]
            mujoco.mj_step(self.model, self.data)
            if on_substep is not None:
                on_substep()

    def base_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.data.xpos[self._body].copy(), self.data.xmat[self._body].reshape(3, 3).copy()

    def base_velocity(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linear velocity in the world frame and angular velocity in the base frame."""
        vx, vy, wz = self.data.qvel[self._dof_adr]
        return np.array([vx, vy, 0.0]), np.array([0.0, 0.0, wz])

    def joint_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.zeros(0), np.zeros(0)

    def part(self, geom: int) -> Part:
        return self._parts[geom]
