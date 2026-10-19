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

"""Alfred in MuJoCo: the CAD model, lift and arms welded, on a planar holonomic base."""

from __future__ import annotations

from collections.abc import Callable
import math

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.msgs.sim_msgs.Contacts import Part
from dimos.robot.diy.alfred.config import alfred_model
from dimos.robot.diy.alfred.mount_tf import mount_transforms
from dimos.simulation.sim_world.world import COLLISION_GROUP, ROBOT_VISUAL_GROUP

CONTROL_DT = 0.02
BASE_BODY = "base_link"
FLOOR_CLEARANCE = 0.01
VELOCITY_TIME_CONSTANT = 0.05
PARTS: dict[str, Part] = {"base": "base", "lift": "lift", "arm": "arm"}
PLANAR_JOINTS = (
    ("x", mujoco.mjtJoint.mjJNT_SLIDE, (1.0, 0.0, 0.0)),
    ("y", mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 1.0, 0.0)),
    ("yaw", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0)),
)


def mid360_mount() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """base_link -> mid360_link as a translation and rotation matrix, as tf publishes it."""
    edges = {t.child_frame_id: t for t in mount_transforms("base_link")}
    matrix = edges["mid360_link"].to_matrix()
    return matrix[:3, 3].copy(), matrix[:3, :3].copy()


def _part(link: str) -> Part:
    if link.startswith("openarm_"):
        return "arm"
    if link == "lift_link":
        return "lift"
    return "base"


def alfred_spec() -> mujoco.MjSpec:
    """The welded model on a planar base with no ground, the lidar blind to its own housing."""
    spec = mujoco.MjSpec.from_string(alfred_model().load().xml)
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE  # type: ignore[attr-defined]
    count: dict[Part, int] = {}
    for body in spec.bodies:
        for geom in body.geoms:
            part = _part(body.name)
            geom.name = f"{part}_{count.setdefault(part, 0)}"
            count[part] += 1
            geom.group = ROBOT_VISUAL_GROUP
    for geom in spec.body(BASE_BODY).geoms:
        sink = FLOOR_CLEARANCE - (geom.pos[2] - geom.size[2])
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX and sink > 0:
            lift = np.array([0.0, 0.0, sink / 2])
            geom.pos = geom.pos + lift
            geom.size = geom.size - lift
    mount, _ = mid360_mount()
    for geom in spec.body(BASE_BODY).geoms:
        w, x, y, z = geom.quat
        local = Rotation.from_quat([x, y, z, w]).inv().apply(mount - geom.pos)
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX and np.all(np.abs(local) <= geom.size):
            geom.group = COLLISION_GROUP
    base = spec.body(BASE_BODY)
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
    return spec


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
    """Alfred in a compiled model, tracking a body-frame velocity command with no wheel slip."""

    dt = CONTROL_DT
    joint_names: tuple[str, ...] = ()

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
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
        self._parts: dict[int, Part] = {}
        for geom in range(model.ngeom):
            part = PARTS.get(model.geom(geom).name.rsplit("_", 1)[0])
            if part is not None:
                self._parts[geom] = part

    def reset(self, x: float, y: float, z_feet: float, yaw: float) -> None:
        """Set base_link on the floor at z_feet, at rest."""
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
