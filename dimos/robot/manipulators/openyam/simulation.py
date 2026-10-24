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

"""Resolved YAM artifacts and native position-servo bindings for sim2.

The supplied artifact owns geometry and servo gains. No hardware driver, policy,
asset downloader or training framework is imported. Public gripper coordinates
are a simulation convention: 0..1 maps to left_finger 0..0.0355 metres.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from dimos.robot.manipulators.openyam.joints import OPENYAM_JOINTS
from dimos.sim2.spec import Joint, RobotConfig, RobotInstance

if TYPE_CHECKING:
    from dimos.sim2.client import SimRobotClient
    from dimos.sim2.messages import StateFrame

MODEL_JOINTS = (*tuple(f"joint{i}" for i in range(1, 7)), "left_finger")
PUBLIC_JOINTS = tuple(OPENYAM_JOINTS)
PASSIVE_JOINT = "right_finger"
GRIPPER_STROKE = 0.0355
TARGET_LIMITS = (
    (-2.60799, 3.04433),
    (0.01, 3.65519),
    (0.01, 3.65519),
    (-1.5608, 1.5608),
    (-1.5608, 1.5608),
    (-2.0844, 2.0844),
    (0.0, GRIPPER_STROKE),
)
CONTRACT_SCHEMA = "humanoid-control-lab.yam-simulated.v1"


class ActuatorContract(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    name: str
    joint: str
    gain: tuple[float, ...]
    bias: tuple[float, ...]
    force_range: tuple[float, float]
    force_limited: bool
    control_limited: bool
    joint_range: tuple[float, float]
    armature: float


class YamContract(BaseModel):
    """Physical fields of the v1 resolved artifact; revisions are provenance only."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    schema_id: str = Field(alias="schema")
    model_joint_names: tuple[str, ...]
    public_joint_names: tuple[str, ...]
    passive_joint_names: tuple[str, ...]
    root_body: str
    grasp_site: str
    initial_positions: dict[str, float]
    initial_root_position: tuple[float, float, float]
    target_limits: dict[str, tuple[float, float]]
    gripper_public_to_model_scale: float
    physics_dt: float = Field(gt=0)
    control_decimation: int = Field(gt=0)
    actuators: tuple[ActuatorContract, ...]
    files: dict[str, str]
    mjlab_revision: str
    dimos_revision: str
    mujoco_version: str
    hardware_calibration: str


def _coordinates(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim == 0 or result.shape[-1] != len(MODEL_JOINTS) or not np.isfinite(result).all():
        raise ValueError("expected finite YAM coordinates with last dimension 7")
    return result


def _validate_targets(values: NDArray[np.float64]) -> None:
    limits = np.asarray(TARGET_LIMITS)
    if np.any(values < limits[:, 0]) or np.any(values > limits[:, 1]):
        raise ValueError("YAM position targets exceed the simulation target limits")


def model_to_public(values: ArrayLike, *, targets: bool = True) -> NDArray[np.float64]:
    """Convert six radians and left-finger metres to the public seven coordinates."""
    result = _coordinates(values)
    if targets:
        _validate_targets(result)
    result[..., -1] /= GRIPPER_STROKE
    return result


def public_to_model(values: ArrayLike, *, targets: bool = True) -> NDArray[np.float64]:
    """Convert public positions; measured feedback need not obey target bounds."""
    result = _coordinates(values)
    result[..., -1] *= GRIPPER_STROKE
    if targets:
        _validate_targets(result)
    return result


def _artifact_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or not path.parts:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"artifact file escapes its folder: {relative!r}")
    return resolved


def _validate_resources(root: Path, contract: YamContract) -> None:
    if "robot.xml" not in contract.files:
        raise ValueError("artifact manifest must include robot.xml")
    for relative, expected in contract.files.items():
        actual = hashlib.sha256(_artifact_file(root, relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact SHA-256 mismatch: {relative}")
    xml = ET.parse(root / "robot.xml").getroot()
    if xml.findall(".//include") or xml.findall(".//extension"):
        raise ValueError("expected resolved MJCF without includes or plugins")
    compiler = xml.find("compiler")
    for element in xml.iter():
        resource = element.get("file")
        if resource is None:
            continue
        directory = ""
        if compiler is not None:
            directory = compiler.get("assetdir", "")
            if element.tag in ("mesh", "texture"):
                directory = compiler.get(f"{element.tag}dir", directory)
        relative = str(PurePosixPath(directory) / resource)
        _artifact_file(root, relative)
        if relative not in contract.files:
            raise ValueError(f"unmanifested MJCF resource: {relative}")


def _validate_model(model: mujoco.MjModel, contract: YamContract) -> None:
    if tuple(model.joint(i).name for i in range(model.njnt)) != (*MODEL_JOINTS, PASSIVE_JOINT):
        raise ValueError("MJCF joint order differs from the YAM contract")
    if tuple(model.actuator(i).name for i in range(model.nu)) != MODEL_JOINTS:
        raise ValueError("MJCF actuator order differs from the YAM contract")
    if model.nq != 8 or model.nv != 8 or model.na:
        raise ValueError("YAM requires eight scalar coordinates and stateless native servos")
    if not np.isclose(model.opt.timestep, contract.physics_dt, rtol=0, atol=1e-12):
        raise ValueError("MJCF timestep differs from its contract")
    root = model.body(contract.root_body)
    if model.body_parentid[root.id] != 0 or model.body_mocapid[root.id] < 0:
        raise ValueError("YAM requires a fixed mocap root attached to the world")
    expected_types = [mujoco.mjtJoint.mjJNT_HINGE] * 6 + [mujoco.mjtJoint.mjJNT_SLIDE] * 2
    if not np.array_equal(model.jnt_type, expected_types):
        raise ValueError("YAM requires six hinge joints and two sliding fingers")
    model.site(contract.grasp_site)
    for index, expected in enumerate(contract.actuators):
        joint = model.joint(expected.joint)
        if expected.name != MODEL_JOINTS[index] or expected.joint != MODEL_JOINTS[index]:
            raise ValueError("contract actuator bindings differ from canonical YAM order")
        if (
            model.actuator_trntype[index] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[index, 0] != joint.id
            or model.actuator_dyntype[index] != mujoco.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[index] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[index] != mujoco.mjtBias.mjBIAS_AFFINE
            or not np.array_equal(model.actuator_gear[index], [1, 0, 0, 0, 0, 0])
        ):
            raise ValueError("YAM requires direct native position servos")
        gain, bias = model.actuator_gainprm[index], model.actuator_biasprm[index]
        if gain[0] <= 0 or bias[0] != 0 or bias[1] != -gain[0] or bias[2] > 0:
            raise ValueError("actuator is not a stable native position servo")
        for actual, declared in (
            (gain, expected.gain),
            (bias, expected.bias),
            (model.actuator_forcerange[index], expected.force_range),
            (model.jnt_range[joint.id], expected.joint_range),
            (model.dof_armature[joint.dofadr[0]], expected.armature),
        ):
            if np.shape(actual) != np.shape(declared) or not np.allclose(
                actual, declared, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(f"MJCF servo parameters differ for {expected.name}")
        if (
            bool(model.actuator_forcelimited[index]) != expected.force_limited
            or bool(model.actuator_ctrllimited[index]) != expected.control_limited
        ):
            raise ValueError(f"MJCF servo limit flags differ for {expected.name}")
        lower, upper = TARGET_LIMITS[index]
        if (
            not model.jnt_limited[joint.id]
            or not expected.joint_range[0] <= lower <= upper <= expected.joint_range[1]
        ):
            raise ValueError(f"target limits escape native joint range: {expected.name}")
        if (
            expected.control_limited
            and not model.actuator_ctrlrange[index, 0]
            <= lower
            <= upper
            <= model.actuator_ctrlrange[index, 1]
        ):
            raise ValueError(f"target limits escape native actuator range: {expected.name}")
    for name, value in contract.initial_positions.items():
        joint = model.joint(name)
        if (
            model.jnt_limited[joint.id]
            and not model.jnt_range[joint.id, 0] <= value <= model.jnt_range[joint.id, 1]
        ):
            raise ValueError(f"initial position escapes native joint range: {name}")
    # Either orientation of the symmetric finger equality is equivalent.
    finger_ids = {model.joint("left_finger").id, model.joint(PASSIVE_JOINT).id}
    coupled = any(
        model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
        and {int(model.eq_obj1id[i]), int(model.eq_obj2id[i])} == finger_ids
        and bool(model.eq_active0[i])
        and np.array_equal(model.eq_data[i, :5], [0, -1, 0, 0, 0])
        for i in range(model.neq)
    )
    if (
        not coupled
        or abs(
            contract.initial_positions["left_finger"] + contract.initial_positions[PASSIVE_JOINT]
        )
        > 1e-12
    ):
        raise ValueError("YAM requires coupled fingers with a consistent initial opening")


@dataclass(frozen=True)
class YamArtifact:
    directory: Path
    contract: YamContract
    contract_sha256: str

    def robot_instance(self) -> RobotInstance:
        """Describe this verified robot without opening a driver or starting physics."""
        joints = tuple(
            Joint(
                name=public,
                model_name=model,
                actuator=model,
                mode="position",
                scale=GRIPPER_STROKE if model == "left_finger" else 1.0,
                ctrl_scale=GRIPPER_STROKE if model == "left_finger" else 1.0,
                lower=0.0 if model == "left_finger" else limits[0],
                upper=1.0 if model == "left_finger" else limits[1],
                max_velocity=1.0 if model == "left_finger" else 0.5,
            )
            for model, public, limits in zip(
                MODEL_JOINTS, PUBLIC_JOINTS, TARGET_LIMITS, strict=True
            )
        )
        initial = model_to_public([self.contract.initial_positions[name] for name in MODEL_JOINTS])
        return RobotInstance(
            RobotConfig(
                self.directory / "robot.xml",
                self.contract.root_body,
                joints,
                max_timestep=self.contract.physics_dt,
            ),
            xyz=self.contract.initial_root_position,
            initial_positions=dict(zip(PUBLIC_JOINTS, initial.tolist(), strict=True)),
            passive_joint_positions={PASSIVE_JOINT: self.contract.initial_positions[PASSIVE_JOINT]},
            controller={
                "kind": "yam-native-position",
                "artifact_contract_sha256": self.contract_sha256,
                "artifact_directory": str(self.directory),
                "hardware_calibration": self.contract.hardware_calibration,
            },
        )


def load_yam_artifact(
    directory: Path, *, expected_contract_sha256: str | None = None
) -> YamArtifact:
    """Validate manifest and compiled physics before returning a reusable descriptor.

    Hashes detect corruption; pin expected_contract_sha256 when the manifest itself
    must be authenticated against a previously selected artifact. Keep the supplied
    folder immutable while a world uses it. Source revisions are retained metadata,
    not requirements that this installation runs the original exporter checkout.
    """
    root = directory.resolve(strict=True)
    raw = _artifact_file(root, "contract.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_contract_sha256 is not None and digest != expected_contract_sha256:
        raise ValueError("artifact contract SHA-256 mismatch")
    contract = YamContract.model_validate_json(raw)
    if (
        contract.schema_id != CONTRACT_SCHEMA
        or contract.model_joint_names != MODEL_JOINTS
        or contract.public_joint_names != PUBLIC_JOINTS
        or contract.passive_joint_names != (PASSIVE_JOINT,)
        or contract.gripper_public_to_model_scale != GRIPPER_STROKE
        or contract.target_limits != dict(zip(MODEL_JOINTS, TARGET_LIMITS, strict=True))
        or set(contract.initial_positions) != {*MODEL_JOINTS, PASSIVE_JOINT}
        or len(contract.actuators) != len(MODEL_JOINTS)
    ):
        raise ValueError("unsupported YAM artifact coordinate contract")
    model_to_public([contract.initial_positions[name] for name in MODEL_JOINTS])
    _validate_resources(root, contract)
    _validate_model(mujoco.MjModel.from_xml_path(str(root / "robot.xml")), contract)
    return YamArtifact(root, contract, digest)


class YamSimAdapter:
    """Coordinate conversion at a caller-owned SimRobotClient boundary."""

    def __init__(self, client: SimRobotClient) -> None:
        if client.descriptor.joint_names != PUBLIC_JOINTS or client.descriptor.actuator_modes != (
            "position",
        ) * len(PUBLIC_JOINTS):
            raise ValueError("client must use the YAM joint order and native position servos")
        self._client = client

    def write_model_targets(self, targets: ArrayLike, *, observed: StateFrame) -> int:
        """Use the policy's actual observation; never refresh its freshness implicitly."""
        public = model_to_public(targets)
        if public.ndim != 1:
            raise ValueError("one client accepts one YAM target vector")
        return self._client.write_positions(public.tolist(), observed=observed)

    def model_positions(self, observed: StateFrame) -> NDArray[np.float64]:
        return public_to_model(observed.values["position"], targets=False)
