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


"""Rerunnable source-to-MuJoCo static audit. This does not validate dynamics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.rl.model import (
    ARM_JOINT_NAMES,
    MANIFEST_PATH,
    load_model,
    source_root,
)


def _rotation(rpy: str) -> NDArray[np.float64]:
    roll, pitch, yaw = np.fromstring(rpy, sep=" ")
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _quaternion_rotation(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)  # type: ignore[attr-defined]  # Compiled API lacks this stub.
    return result.reshape(3, 3)


def audit_model(path: str | Path | None = None) -> dict[str, Any]:
    """Assert per-link source transfer and return compact numerical evidence.

    Tensor Frobenius relative tolerance 3e-5 accounts for XML decimal serialization.
    Absolute tolerances cover small off-diagonal tensor entries and transforms.
    """
    root = source_root(path)
    manifest = json.loads(MANIFEST_PATH.read_text())
    urdf = ET.parse(root / manifest["urdf"]).getroot()
    model = load_model(root)
    links = []
    for link in urdf.findall("link"):
        name = link.attrib["name"]
        body_id = model.body(name).id
        inertial = link.find("inertial")
        assert inertial is not None
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        origin = inertial.find("origin")
        assert mass is not None and inertia is not None and origin is not None
        values = {key: float(value) for key, value in inertia.attrib.items()}
        tensor = np.array(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ]
        )
        rotation = _rotation(origin.attrib.get("rpy", "0 0 0"))
        source_tensor = rotation @ tensor @ rotation.T
        eigenvalues = np.linalg.eigvalsh(source_tensor)
        assert eigenvalues[0] > 0, f"Nonpositive source inertia: {name}"
        assert eigenvalues[2] <= eigenvalues[:2].sum() + 1e-12, f"Invalid inertia triangle: {name}"
        imported_rotation = _quaternion_rotation(model.body_iquat[body_id])
        imported_tensor = (
            imported_rotation @ np.diag(model.body_inertia[body_id]) @ imported_rotation.T
        )
        relative_tensor_error = float(
            np.linalg.norm(imported_tensor - source_tensor) / np.linalg.norm(source_tensor)
        )
        assert relative_tensor_error < 3e-5, f"Full inertia tensor transfer differs: {name}"
        source_mass = float(mass.attrib["value"])
        np.testing.assert_allclose(
            model.body_mass[body_id], source_mass, rtol=3e-5, atol=1e-10, err_msg=name
        )
        np.testing.assert_allclose(
            model.body_ipos[body_id],
            np.fromstring(origin.attrib["xyz"], sep=" "),
            rtol=3e-5,
            atol=1e-8,
            err_msg=name,
        )
        geoms = np.flatnonzero(model.geom_bodyid == body_id)
        for element, visual in (("visual", True), ("collision", False)):
            expected = sorted(
                Path(mesh.attrib["filename"]).stem
                for mesh in link.findall(f"{element}/geometry/mesh")
            )
            selected = [int(g) for g in geoms if (model.geom_group[g] == 1) == visual]
            actual = sorted(model.mesh(int(model.geom_dataid[g])).name for g in selected)
            assert actual == expected, f"Mesh mapping differs for {name} {element}"
            for geom_id in selected:
                assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
                expected_mask = 0 if visual else 1
                assert model.geom_contype[geom_id] == expected_mask
                assert model.geom_conaffinity[geom_id] == expected_mask
        links.append(
            {
                "name": name,
                "mass_kg": source_mass,
                "inertia_eigenvalues_kg_m2": eigenvalues.tolist(),
                "relative_tensor_error": relative_tensor_error,
                "max_tensor_error_kg_m2": float(np.max(np.abs(imported_tensor - source_tensor))),
                "max_com_error_m": float(
                    np.max(
                        np.abs(
                            model.body_ipos[body_id] - np.fromstring(origin.attrib["xyz"], sep=" ")
                        )
                    )
                ),
            }
        )
    transforms = []
    for joint in urdf.findall("joint"):
        origin, child, parent = joint.find("origin"), joint.find("child"), joint.find("parent")
        assert origin is not None and child is not None and parent is not None
        body_id = model.body(child.attrib["link"]).id
        assert model.body_parentid[body_id] == model.body(parent.attrib["link"]).id
        expected_position = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        np.testing.assert_allclose(model.body_pos[body_id], expected_position, rtol=3e-5, atol=1e-8)
        expected_rotation = _rotation(origin.attrib.get("rpy", "0 0 0"))
        np.testing.assert_allclose(
            _quaternion_rotation(model.body_quat[body_id]), expected_rotation, atol=2e-6
        )
        transforms.append(joint.attrib["name"])
    expected_exclusions = {
        tuple(sorted((model.body(a).id, model.body(b).id)))
        for a, b in manifest["collision_exclusions"]
    }
    actual_exclusions = {
        (int(signature) >> 16, int(signature) & 65535) for signature in model.exclude_signature
    }
    assert actual_exclusions == expected_exclusions
    actuators = []
    for name in ARM_JOINT_NAMES:
        actuator_id = model.actuator(name + "_position").id
        joint_id = model.joint(name).id
        source_joint = urdf.find(f"joint[@name='{name}']")
        assert source_joint is not None
        limit = source_joint.find("limit")
        assert limit is not None
        effort = float(limit.attrib["effort"])
        assert model.actuator_trnid[actuator_id, 0] == joint_id
        assert model.actuator_ctrllimited[actuator_id] and model.actuator_forcelimited[actuator_id]
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator_id],
            [float(limit.attrib["lower"]), float(limit.attrib["upper"])],
            atol=5e-6,
        )
        np.testing.assert_allclose(model.actuator_forcerange[actuator_id], [-effort, effort])
        np.testing.assert_allclose(model.actuator_gainprm[actuator_id, :3], [80, 0, 0])
        np.testing.assert_allclose(model.actuator_biasprm[actuator_id, :3], [0, -80, -8])
        assert model.dof_damping[model.jnt_dofadr[joint_id]] == 2
        actuators.append(
            {
                "joint": name,
                "effort_cap_nm": effort,
                "imported_force_range_nm": model.actuator_forcerange[actuator_id].tolist(),
                "imported_control_range_rad": model.actuator_ctrlrange[actuator_id].tolist(),
                "imported_gain_prm": model.actuator_gainprm[actuator_id, :3].tolist(),
                "imported_bias_prm": model.actuator_biasprm[actuator_id, :3].tolist(),
                "imported_damping": float(model.dof_damping[model.jnt_dofadr[joint_id]]),
                "source_velocity_metadata_rad_s": float(limit.attrib["velocity"]),
            }
        )
    assert model.opt.timestep == 0.002
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # type: ignore[attr-defined]  # Compiled enum lacks this stub.
    return {
        "status": "static_import_checks_passed",
        "mujoco_version": importlib.metadata.version("mujoco"),
        "source_commit": manifest["commit"],
        "links": links,
        "joint_origin_transforms_checked": transforms,
        "collision_geometries": int(np.sum(model.geom_contype != 0)),
        "noncolliding_visual_geometries": int(np.sum(model.geom_group == 1)),
        "exclusion_pairs_checked": len(actual_exclusions),
        "actuators": actuators,
        "simulation_assumptions": {
            "kp": 80,
            "kv": 8,
            "joint_damping": 2,
            "timestep_s": 0.002,
            "integrator": "implicitfast",
            "fixed_base_and_torso": True,
        },
        "limitations": [
            "Convex mesh collision does not reproduce concave STL geometry.",
            "URDF velocity metadata is not a MuJoCo hard speed constraint.",
            "Mass and inertia transfer checks do not validate real dynamics.",
            "Controller gains and damping are provisional simulation parameters.",
        ],
        "hardware_variant_verified": False,
        "dynamics_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = audit_model(args.source_root)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Static import audit passed for {len(evidence['links'])} links; evidence: {args.output}")


if __name__ == "__main__":
    main()
