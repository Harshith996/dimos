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

"""Known YAM geometry and kinematics, independent of the simulated scene.

Frame ledger: metres/radians; world X forward, Y left, Z up. The two fixed
bases are 0.62 m apart (an example layout, not an office calibration). Each
link frame, joint axis and grasp frame comes from the validated robot MJCF.
The IK-only URDF retains six hinge coordinates per arm; finger bodies are
fixed because they do not change the grasp site. It has no dynamics or scene
geometry. Display meshes use the same MJCF and only measured joint positions.
"""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF  # type: ignore[import-untyped]

from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_SIDES
from dimos.robot.manipulators.openyam.simulation import (
    GRIPPER_STROKE,
    MODEL_JOINTS,
    TARGET_LIMITS,
    YamArtifact,
)

BASE_POSITIONS = {"left": (0.0, 0.31, 0.0), "right": (0.0, -0.31, 0.0)}


def artifact_directory() -> Path:
    """One deployment setting shared by physics, control and known robot geometry."""
    return Path(
        os.environ.get("DIMOS_YAM_ARTIFACT", str(Path.home() / ".local/share/dimos/yam-sim2-model"))
    )


def _numbers(values: Any) -> str:
    return " ".join(format(float(value), ".16g") for value in values)


def _origin(parent: ET.Element, xyz: Any, quat: Any) -> None:
    rpy = Rotation.from_quat(np.roll(quat, -1)).as_euler("xyz")
    ET.SubElement(parent, "origin", xyz=_numbers(xyz), rpy=_numbers(rpy))


def write_ik_model(artifact: YamArtifact, destination: Path) -> Path:
    """Derive a frame-only URDF; fail on unsupported source joint semantics."""
    model = mujoco.MjModel.from_xml_path(str(artifact.directory / "robot.xml"))
    root = ET.Element("robot", name="dual_yam_artifact_kinematics")
    ET.SubElement(root, "link", name="world")
    for side in DUAL_OPENYAM_SIDES:
        for body in range(1, model.nbody):
            name = f"{side}_{model.body(body).name}"
            parent_id = int(model.body_parentid[body])
            parent = f"{side}_{model.body(parent_id).name}" if parent_id else "world"
            ET.SubElement(root, "link", name=name)
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError("YAM IK export expects at most one scalar joint per body")
            joint_id = int(model.body_jntadr[body]) if count else -1
            native_name = model.joint(joint_id).name if count else ""
            moving = native_name in MODEL_JOINTS[:6]
            if moving and (
                model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
                or np.any(model.jnt_pos[joint_id] != 0)
            ):
                raise ValueError("YAM IK export requires hinge axes at body origins")
            joint = ET.SubElement(
                root,
                "joint",
                name=f"{side}_{native_name}" if moving else f"{name}_fixed",
                type="revolute" if moving else "fixed",
            )
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            _origin(
                joint,
                model.body_pos[body] if parent_id else BASE_POSITIONS[side],
                model.body_quat[body],
            )
            if moving:
                index = MODEL_JOINTS.index(native_name)
                ET.SubElement(joint, "axis", xyz=_numbers(model.jnt_axis[joint_id]))
                ET.SubElement(
                    joint,
                    "limit",
                    lower=str(TARGET_LIMITS[index][0]),
                    upper=str(TARGET_LIMITS[index][1]),
                    effort="100",
                    velocity="2",
                )
        site = model.site(artifact.contract.grasp_site)
        name = f"{side}_grasp_frame"
        ET.SubElement(root, "link", name=name)
        joint = ET.SubElement(root, "joint", name=f"{name}_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=f"{side}_{model.body(site.bodyid[0]).name}")
        ET.SubElement(joint, "child", link=name)
        _origin(joint, site.pos, site.quat)
    ET.indent(root)
    destination.write_text(ET.tostring(root, encoding="unicode"))
    return destination


class KnownYamModel:
    """Forward kinematics of the robot asset; never reads SimulationModule/world truth."""

    def __init__(self, artifact: YamArtifact) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(artifact.directory / "robot.xml"))
        self.data = {side: mujoco.MjData(self.model) for side in DUAL_OPENYAM_SIDES}
        self.grasp_site = artifact.contract.grasp_site
        self.geoms = [i for i in range(self.model.ngeom) if self.model.geom_group[i] == 2]
        self.geometry = self._geometry()

    def _geometry(self) -> dict[str, Any]:
        model = self.model
        meshes: dict[str, Any] = {}
        geometries = []
        for i in self.geoms:
            if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("Expected mesh-only YAM visual geometry")
            mesh = int(model.geom_dataid[i])
            if str(mesh) not in meshes:
                v, n = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                f, k = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                meshes[str(mesh)] = {
                    "vertices": self.model.mesh_vert[v : v + n].ravel().tolist(),
                    "faces": self.model.mesh_face[f : f + k].ravel().tolist(),
                }
            mat = int(model.geom_matid[i])
            color = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[i]
            geometries.append({"id": i, "mesh": str(mesh), "rgba": color.tolist()})
        return {
            "meshes": meshes,
            "arms": {side: geometries for side in DUAL_OPENYAM_SIDES},
            "sides": list(DUAL_OPENYAM_SIDES),
            "finger_geometry": True,
        }

    def feedback(self, positions: dict[str, float]) -> dict[str, Any]:
        arms = {}
        for side, data in self.data.items():
            for i, name in enumerate(MODEL_JOINTS[:6]):
                data.joint(name).qpos[0] = positions[f"{side}_joint{i + 1}"]
            displacement = positions[f"{side}_arm/gripper"] * GRIPPER_STROKE
            data.joint("left_finger").qpos[0] = displacement
            data.joint("right_finger").qpos[0] = -displacement
            data.mocap_pos[0] = BASE_POSITIONS[side]
            mujoco.mj_forward(self.model, data)
            transforms = []
            for geom in self.geoms:
                matrix = np.eye(4)
                matrix[:3, :3] = data.geom_xmat[geom].reshape(3, 3)
                matrix[:3, 3] = data.geom_xpos[geom]
                transforms.append(matrix.T.ravel().tolist())  # Three.js column-major
            site = data.site(self.grasp_site)
            arms[side] = {
                "transforms": transforms,
                "grasp": site.xpos.tolist(),
                "orientation": Rotation.from_matrix(site.xmat.reshape(3, 3)).as_quat().tolist(),
                "opening": float(np.clip(positions[f"{side}_arm/gripper"], 0, 1)),
            }
        return arms


class KnownUrdfYamModel:
    """Display the same resolved URDF and base pose used by physical-arm IK.

    The arm-only hardware URDF fixes its fingertips. Hide those fixed visuals;
    depth observations and the measured opening gauge represent the jaws until
    a calibrated moving-finger description is supplied. Never guess their axes.
    """

    def __init__(self, config: RobotModelConfig) -> None:
        loaded = config.model.load()
        self.robot = URDF.load(BytesIO(loaded.xml.encode()), load_collision_meshes=False)
        self.base = np.eye(4)
        self.base[:3, :3] = Rotation.from_quat(config.base_pose.orientation.to_list()).as_matrix()
        self.base[:3, 3] = config.base_pose.position.to_list()
        self.nodes: dict[str, list[str]] = {side: [] for side in DUAL_OPENYAM_SIDES}
        meshes: dict[str, Any] = {}
        arms: dict[str, list[dict[str, Any]]] = {side: [] for side in DUAL_OPENYAM_SIDES}
        fixed_tips = {
            joint.child_link
            for joint in loaded.joints
            if joint.type == "fixed" and joint.child_link.endswith(("_tip_left", "_tip_right"))
        }
        for parent, node, edge in self.robot.scene.graph.to_edgelist():
            name = edge.get("geometry")
            side = next((s for s in DUAL_OPENYAM_SIDES if parent.startswith(f"{s}_")), None)
            if name is None or side is None or parent in fixed_tips:
                continue
            mesh = self.robot.scene.geometry[name]
            meshes[name] = {
                "vertices": mesh.vertices.ravel().tolist(),
                "faces": mesh.faces.ravel().tolist(),
            }
            color = np.asarray(mesh.visual.main_color, dtype=float) / 255
            arms[side].append({"mesh": name, "rgba": color.tolist()})
            self.nodes[side].append(node)
        self.geometry = {
            "meshes": meshes,
            "arms": arms,
            "sides": list(DUAL_OPENYAM_SIDES),
            "finger_geometry": not fixed_tips,
        }

    def feedback(self, positions: dict[str, float]) -> dict[str, Any]:
        self.robot.update_cfg(
            {name: positions[name] for name in self.robot.actuated_joint_names if name in positions}
        )
        arms = {}
        for side in DUAL_OPENYAM_SIDES:
            grasp = self.base @ self.robot.get_transform(f"{side}_grasp_frame")
            arms[side] = {
                "transforms": [
                    (self.base @ self.robot.get_transform(node)).T.ravel().tolist()
                    for node in self.nodes[side]
                ],
                "grasp": grasp[:3, 3].tolist(),
                "orientation": Rotation.from_matrix(grasp[:3, :3]).as_quat().tolist(),
                "opening": float(np.clip(positions[f"{side}_arm/gripper"], 0, 1)),
            }
        return arms
