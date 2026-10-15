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

"""Per-joint tf for an articulated robot, posed by forward kinematics from live joint angles."""

from __future__ import annotations

from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.assets.model import RobotModel
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ArticulatedTf:
    """FK for every URDF joint, one parent -> child edge each; pinocchio because yourdfpy is absent on aarch64."""

    def __init__(self, model: RobotModel, root_link: str) -> None:
        import xml.etree.ElementTree as ElementTree

        import pinocchio

        loaded = model.load()
        self._model = pinocchio.buildModelFromXML(loaded.xml)
        self._data = self._model.createData()
        self._neutral_q = pinocchio.neutral(self._model)
        self._unknown_joints: set[str] = set()

        self._joint_q_index = {
            self._model.names[joint_id]: self._model.joints[joint_id].idx_q
            for joint_id in range(1, self._model.njoints)
            if self._model.joints[joint_id].nq == 1
        }
        rename = {loaded.root_link: root_link}
        self._edges: list[tuple[str, str, int, int]] = []
        for joint in ElementTree.fromstring(loaded.xml).iter("joint"):
            parent = joint.find("parent").attrib["link"]  # type: ignore[union-attr]
            child = joint.find("child").attrib["link"]  # type: ignore[union-attr]
            self._edges.append(
                (
                    rename.get(parent, parent),
                    child,
                    self._model.getFrameId(parent),
                    self._model.getFrameId(child),
                )
            )

    def transforms(self, joint_state: JointState) -> list[Transform]:
        import pinocchio
        from scipy.spatial.transform import Rotation

        q = self._neutral_q.copy()
        for name, position in zip(joint_state.name, joint_state.position, strict=False):
            index = self._joint_q_index.get(name)
            if index is None:
                if name not in self._unknown_joints:
                    self._unknown_joints.add(name)
                    logger.warning("FK: no model joint named %r", name)
                continue
            q[index] = position

        pinocchio.framesForwardKinematics(self._model, self._data, q)
        # Fixed joints are republished with the moving ones: a chained lookup is only as fresh as its stalest edge.
        placements = [
            self._data.oMf[parent_id].actInv(self._data.oMf[child_id])
            for _, _, parent_id, child_id in self._edges
        ]
        quaternions = Rotation.from_matrix([p.rotation for p in placements]).as_quat()
        return [
            Transform(
                translation=Vector3(*placement.translation),
                rotation=Quaternion(*quaternion),
                frame_id=parent,
                child_frame_id=child,
                ts=joint_state.ts,
            )
            for (parent, child, _, _), placement, quaternion in zip(
                self._edges, placements, quaternions, strict=True
            )
        ]
