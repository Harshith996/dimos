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

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.msgs.sim_msgs.Contacts import Part
from dimos.robot.diy.alfred.config import alfred_model
from dimos.robot.diy.alfred.mount_tf import mount_transforms
from dimos.simulation.sim_world.planar import add_planar_joints
from dimos.simulation.sim_world.world import COLLISION_GROUP, ROBOT_VISUAL_GROUP

BASE_BODY = "base_link"
FLOOR_CLEARANCE = 0.01
PARTS: tuple[Part, ...] = ("base", "lift", "arm")


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
    add_planar_joints(spec, BASE_BODY)
    return spec
