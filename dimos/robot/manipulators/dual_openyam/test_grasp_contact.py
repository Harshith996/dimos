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

"""The pick's grasp contact check on the dual OpenYAM workcell model.

Grasps sit where the Spam can stood in SceneReplica scene 56, in the right arm's reach.
The model is dual_openyam_workcell_v5: prismatic fingers, exact fingertip meshes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.spatial.transform import Rotation

from dimos.manipulation.planning.factory import create_world
from dimos.manipulation.planning.gripper_contact import GraspContactChecker
from dimos.manipulation.planning.spec.validation import prepare_robot_model
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.dual_openyam.blueprints.grasp import (
    DUAL_OPENYAM_TABLE_TOP_Z,
    dual_openyam_contact_model_config,
    dual_openyam_grasp_model_config,
)

CONTACT_MARGIN = 0.005
OBSTACLE_MARGIN = 0.003
X, Y, HEIGHT = 0.20, -0.09, 0.08
TOP_DOWN = Rotation.from_euler("xyz", [np.pi, 0.0, 0.0])


def _pose(x: float, y: float, z: float, rotation: Rotation = TOP_DOWN) -> Pose:
    q = rotation.as_quat()
    return Pose(position=Vector3(x=x, y=y, z=z), orientation=Quaternion(q[0], q[1], q[2], q[3]))


def _box(center: tuple[float, float, float], size: tuple[float, float, float]) -> NDArray:
    """Points 4 mm apart on the faces of an axis-aligned box."""
    half = np.asarray(size) / 2.0
    faces = []
    for axis in range(3):
        u, v = (a for a in range(3) if a != axis)
        grid_u = np.arange(-half[u], half[u] + 1e-9, 0.004)
        grid_v = np.arange(-half[v], half[v] + 1e-9, 0.004)
        mesh_u, mesh_v = np.meshgrid(grid_u, grid_v)
        for side in (-1.0, 1.0):
            face = np.zeros((mesh_u.size, 3))
            face[:, u], face[:, v], face[:, axis] = (
                mesh_u.ravel(),
                mesh_v.ravel(),
                side * half[axis],
            )
            faces.append(face)
    return np.concatenate(faces) + np.asarray(center)


def _table(hole: float) -> NDArray:
    """Table points 1 cm apart, minus those within ``hole`` of the object's spot."""
    x, y = np.meshgrid(np.arange(-0.1, 0.5, 0.01), np.arange(-0.4, 0.4, 0.01))
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, DUAL_OPENYAM_TABLE_TOP_Z)], axis=1)
    under = (np.abs(points[:, 0] - X) < hole) & (np.abs(points[:, 1] - Y) < hole)
    return points[~under]


def _object(width: float, height: float = HEIGHT) -> NDArray:
    return _box((X, Y, DUAL_OPENYAM_TABLE_TOP_Z + height / 2), (width, width, height))


CAN = _object(0.05)
TABLE = _table(hole=0.03)
MID = DUAL_OPENYAM_TABLE_TOP_Z + HEIGHT / 2


@pytest.fixture(scope="module")
def checker() -> GraspContactChecker:
    return GraspContactChecker(dual_openyam_contact_model_config(), "right_manipulator")


def _clear(checker: GraspContactChecker, grasp: Pose, target: NDArray, *scene: NDArray) -> bool:
    points = np.concatenate([np.empty((0, 3)), *scene])
    return checker.check([grasp], target, points, "world", CONTACT_MARGIN, OBSTACLE_MARGIN)[0]


def test_planner_world_keeps_the_fingers_closed_and_home_collision_free() -> None:
    config = dual_openyam_grasp_model_config()
    assert config.gripper is not None
    assert not set(config.gripper.finger_joints) & set(config.joint_names)
    world = create_world(backend="roboplan")
    world.load_model(prepare_robot_model(config))
    world.finalize()
    home = JointState(name=list(config.joint_names), position=list(config.home_joints or []))
    assert world.check_config_collision_free(home)


def test_keeps_a_top_down_grasp_at_mid_height(checker: GraspContactChecker) -> None:
    assert _clear(checker, _pose(X, Y, MID), CAN, TABLE)


def test_rejects_fingers_closing_on_nothing_above_the_object(checker: GraspContactChecker) -> None:
    assert not _clear(checker, _pose(X, Y, DUAL_OPENYAM_TABLE_TOP_Z + HEIGHT + 0.02), CAN, TABLE)


def test_rejects_fingertips_pressed_into_the_table(checker: GraspContactChecker) -> None:
    assert not _clear(checker, _pose(X, Y, DUAL_OPENYAM_TABLE_TOP_Z - 0.005), CAN, TABLE)


def test_rejects_a_sideways_scoop_along_the_table(checker: GraspContactChecker) -> None:
    scoop = _pose(
        X, Y, DUAL_OPENYAM_TABLE_TOP_Z + 0.02, Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0])
    )
    assert not _clear(checker, scoop, CAN, TABLE)


def test_rejects_a_neighbour_the_open_finger_would_land_on(checker: GraspContactChecker) -> None:
    # The open fingers stand 47 mm from the centre: a neighbour starting 45 mm out is hit,
    # one starting 60 mm out is cleared because the finger tapers above it.
    close = _box((X, Y - 0.06, DUAL_OPENYAM_TABLE_TOP_Z + 0.03), (0.04, 0.03, 0.06))
    clear = _box((X, Y - 0.075, DUAL_OPENYAM_TABLE_TOP_Z + 0.03), (0.04, 0.03, 0.06))
    assert not _clear(checker, _pose(X, Y, MID), CAN, TABLE, close)
    assert _clear(checker, _pose(X, Y, MID), CAN, TABLE, clear)


def test_rejects_what_the_arm_cannot_reach(checker: GraspContactChecker) -> None:
    assert not _clear(checker, _pose(0.95, Y, MID), CAN, TABLE)


def test_open_fingers_clear_objects_up_to_the_jaw_span(checker: GraspContactChecker) -> None:
    # 94 mm between the open pads: a 9 cm object fits at full finger depth, 10 cm does not.
    top = DUAL_OPENYAM_TABLE_TOP_Z + HEIGHT
    assert _clear(checker, _pose(X, Y, top - 0.04), _object(0.09), _table(hole=0.055))
    assert not _clear(checker, _pose(X, Y, top - 0.01), _object(0.10), _table(hole=0.06))


def test_rejects_a_grasp_that_puts_the_finger_carriage_into_a_tall_object(
    checker: GraspContactChecker,
) -> None:
    # The carriages sit 7 to 9 cm above the fingertips and reach the centre when open.
    tall = _object(0.06, height=0.12)
    top = DUAL_OPENYAM_TABLE_TOP_Z + 0.12
    assert _clear(checker, _pose(X, Y, top - 0.04), tall, _table(hole=0.04))
    assert not _clear(checker, _pose(X, Y, top - 0.08), tall, _table(hole=0.04))


def test_the_modelled_slab_swallows_the_camera_table_but_the_contact_model_does_not(
    checker: GraspContactChecker,
) -> None:
    # The overhead D405 puts the table 3 to 15 mm under the fitted top; the slab top is 10 mm
    # under it. With the workcell in the model every grasp collides, wherever the gripper is.
    low_table = _table(hole=0.03) - np.array([0.0, 0.0, 0.012])
    with_slab = GraspContactChecker(dual_openyam_grasp_model_config(), "right_manipulator")
    assert not _clear(with_slab, _pose(X, Y, MID), CAN, low_table)
    assert _clear(checker, _pose(X, Y, MID), CAN, low_table)
