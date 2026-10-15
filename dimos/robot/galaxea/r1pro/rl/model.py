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


"""Pinned R1 Pro geometry for simulation-only arm tracking experiments.

This converter preserves vendor inertials and collision meshes. MuJoCo uses
convex mesh collision, so a successful import does not validate contact dynamics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from numpy.typing import NDArray

ARM_JOINT_NAMES = tuple(f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8))
WRIST_SITE_NAMES = ("left_wrist", "right_wrist")
MANIFEST_PATH = Path(__file__).resolve().parents[5] / "configs/r1pro_rl_assets.json"
DEFAULT_SOURCE_ROOT = (
    Path.home()
    / ".cache/dimos/robot_assets/sources/f15d4276801dd90e/URDF/R1Pro/urdf_r1pro_g1z_2026"
)


def source_root(path: str | Path | None = None) -> Path:
    """Resolve a local package and verify every pinned input. Never downloads."""
    root = Path(
        path if path is not None else os.environ.get("R1PRO_ASSET_ROOT", str(DEFAULT_SOURCE_ROOT))
    )
    manifest = json.loads(MANIFEST_PATH.read_text())
    for relative, expected in manifest["sha256"].items():
        asset = root / relative
        if not asset.is_file():
            raise FileNotFoundError(
                f"Missing pinned R1 Pro asset: {asset}. Set R1PRO_ASSET_ROOT to the package at commit {manifest['commit']}."
            )
        if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            raise ValueError(f"R1 Pro asset differs from pinned manifest: {asset}")
    return root


def build_xml(path: str | Path | None = None) -> str:
    """Build fixed-base/fixed-torso MJCF with 14 simulation position actuators.

    Gains, damping, and the 2 ms timestep are debugging defaults, not identified
    hardware parameters. The wrist sites use gripper origins, not measured TCPs.
    """
    root = source_root(path)
    manifest = json.loads(MANIFEST_PATH.read_text())
    urdf = ET.parse(root / manifest["urdf"]).getroot()
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str(root / "meshes" / Path(mesh.attrib["filename"]).name))
    for joint in urdf.findall("joint"):
        if joint.attrib["name"] not in ARM_JOINT_NAMES:
            joint.set("type", "fixed")
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(
        extension, "compiler", fusestatic="false", discardvisual="false", strippath="false"
    )
    with tempfile.TemporaryDirectory(prefix="r1pro-model-") as directory:
        imported = Path(directory) / "model.urdf"
        ET.ElementTree(urdf).write(imported)
        model = mujoco.MjModel.from_xml_path(str(imported))
        mjcf_path = Path(directory) / "model.xml"
        mujoco.mj_saveLastXML(str(mjcf_path), model)  # type: ignore[attr-defined]  # Compiled MuJoCo API lacks this stub.
        mjcf = ET.parse(mjcf_path).getroot()
    ET.SubElement(mjcf, "option", timestep="0.002", integrator="implicitfast")
    for joint in mjcf.findall(".//joint"):
        joint.set("damping", "2")
    contact = ET.SubElement(mjcf, "contact")
    for first, second in manifest["collision_exclusions"]:
        ET.SubElement(contact, "exclude", body1=first, body2=second)
    actuator = ET.SubElement(mjcf, "actuator")
    joints = {joint.attrib["name"]: joint for joint in mjcf.findall(".//joint")}
    original_joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
    for name in ARM_JOINT_NAMES:
        limit = original_joints[name].find("limit")
        assert limit is not None
        effort = float(limit.attrib["effort"])
        ET.SubElement(
            actuator,
            "position",
            name=f"{name}_position",
            joint=name,
            kp="80",
            kv="8",
            ctrllimited="true",
            ctrlrange=joints[name].attrib["range"],
            forcelimited="true",
            forcerange=f"{-effort} {effort}",
        )
    for side, site_name in zip(("left", "right"), WRIST_SITE_NAMES, strict=True):
        body = mjcf.find(f".//body[@name='{side}_gripper_link']")
        assert body is not None
        ET.SubElement(body, "site", name=site_name, pos="0 0 0", size="0.005", rgba="0 1 0 1")
    return ET.tostring(mjcf, encoding="unicode")


def load_model(path: str | Path | None = None) -> mujoco.MjModel:
    """Return the checked fixed-base, fixed-torso simulation model."""
    return mujoco.MjModel.from_xml_string(build_xml(path))


def joint_addresses(model: mujoco.MjModel) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Return qpos and qvel addresses in left-then-right arm order."""
    ids = np.array([model.joint(name).id for name in ARM_JOINT_NAMES])
    return model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()


def rest_qpos(model: mujoco.MjModel) -> NDArray[np.float64]:
    """Return the vendor zero pose. This does not certify collision clearance."""
    return model.qpos0.copy()


def collision_pairs(
    model: mujoco.MjModel, data: mujoco.MjData, penetration_tolerance_m: float = 1e-5
) -> tuple[tuple[str, str], ...]:
    """List distinct contacting body pairs deeper than numerical tolerance.

    Call mj_forward or mj_step first. This is a mesh-convex collision check,
    not a swept-volume or hardware clearance certificate.
    """
    if not np.isfinite(penetration_tolerance_m) or penetration_tolerance_m < 0:
        raise ValueError("Collision penetration tolerance must be finite and nonnegative")
    pairs = set()
    for contact in data.contact:
        if contact.dist < -penetration_tolerance_m:
            first = model.body(int(model.geom_bodyid[contact.geom1])).name
            second = model.body(int(model.geom_bodyid[contact.geom2])).name
            pairs.add(tuple(sorted((first, second))))
    return tuple(sorted(pairs))
