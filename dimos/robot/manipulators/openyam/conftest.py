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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import pytest

from dimos.robot.manipulators.openyam.simulation import (
    CONTRACT_SCHEMA,
    GRIPPER_STROKE,
    MODEL_JOINTS,
    PASSIVE_JOINT,
    PUBLIC_JOINTS,
    TARGET_LIMITS,
)


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """Independent native model, no meshes, training checkout or GPU dependencies."""
    arm = []
    for index, (name, limits) in enumerate(
        zip(MODEL_JOINTS[:6], TARGET_LIMITS[:6], strict=True), 1
    ):
        arm.append(
            f'<body name="link_{index}" pos="0 0 .05"><joint name="{name}" range="{limits[0] - 0.01} {limits[1] + 0.01}"/><geom type="sphere" size=".01" mass=".1"/>'
        )
    fingers = '<site name="grasp_site"/><body name="left"><joint name="left_finger" type="slide" range="-.01 .05"/><geom type="sphere" size=".01" mass=".01"/></body><body name="right"><joint name="right_finger" type="slide" range="-.05 .01"/><geom type="sphere" size=".01" mass=".01"/></body>'
    actuators = "".join(
        f'<position name="{name}" joint="{name}" kp="20" kv="2" forcerange="-30 30"/>'
        for name in MODEL_JOINTS
    )
    xml = f'<mujoco><compiler angle="radian"/><option timestep=".005" gravity="0 0 0"/><default><joint armature=".1"/><geom contype="0" conaffinity="0"/></default><worldbody><body name="base" mocap="true">{"".join(arm)}{fingers}{"</body>" * 6}</body></worldbody><equality><joint joint1="left_finger" joint2="right_finger" polycoef="0 -1 0 0 0"/></equality><actuator>{actuators}</actuator></mujoco>'
    (tmp_path / "robot.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "robot.xml"))
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "model_joint_names": MODEL_JOINTS,
        "public_joint_names": PUBLIC_JOINTS,
        "passive_joint_names": [PASSIVE_JOINT],
        "root_body": "base",
        "grasp_site": "grasp_site",
        "initial_positions": dict(
            zip(
                (*MODEL_JOINTS, PASSIVE_JOINT),
                [0, 1, 1, -0.9, 0, 0, 0.01875, -0.01875],
                strict=True,
            )
        ),
        "initial_root_position": [0, 0, 0.01],
        "target_limits": dict(zip(MODEL_JOINTS, TARGET_LIMITS, strict=True)),
        "gripper_public_to_model_scale": GRIPPER_STROKE,
        "physics_dt": 0.005,
        "control_decimation": 4,
        "mjlab_revision": "fixture-not-mjlab",
        "dimos_revision": "fixture-not-a-revision",
        "mujoco_version": mujoco.__version__,
        "hardware_calibration": "unverified",
        "files": {"robot.xml": hashlib.sha256((tmp_path / "robot.xml").read_bytes()).hexdigest()},
        "actuators": [
            {
                "name": name,
                "joint": name,
                "gain": model.actuator_gainprm[i].tolist(),
                "bias": model.actuator_biasprm[i].tolist(),
                "force_range": model.actuator_forcerange[i].tolist(),
                "force_limited": bool(model.actuator_forcelimited[i]),
                "control_limited": bool(model.actuator_ctrllimited[i]),
                "joint_range": model.jnt_range[i].tolist(),
                "armature": float(model.dof_armature[i]),
            }
            for i, name in enumerate(MODEL_JOINTS)
        ],
    }
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    return tmp_path
