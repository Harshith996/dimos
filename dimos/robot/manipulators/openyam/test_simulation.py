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

import numpy as np
import pytest
from pytest_mock import MockerFixture

from dimos.robot.manipulators.openyam.demo_sim2 import run_demo
from dimos.robot.manipulators.openyam.simulation import (
    GRIPPER_STROKE,
    PASSIVE_JOINT,
    PUBLIC_JOINTS,
    YamSimAdapter,
    load_yam_artifact,
    model_to_public,
    public_to_model,
)
from dimos.sim2.client import SimRobotClient
from dimos.sim2.messages import FrameMetadata, StateFrame, StreamDescriptor


def test_coordinate_conversion_preserves_arm_and_gripper_units() -> None:
    native = np.array([[0, 1, 2, -0.5, 0.5, 0, 0], [1, 2, 3, 0.5, -0.5, 0, GRIPPER_STROKE]])
    public = model_to_public(native)
    np.testing.assert_array_equal(public[:, :6], native[:, :6])
    np.testing.assert_array_equal(public[:, 6], [0, 1])
    np.testing.assert_allclose(public_to_model(public), native)
    np.testing.assert_array_equal(native[:, 6], [0, GRIPPER_STROKE])


@pytest.mark.parametrize(
    "value", [[0, 1, 1, 0, 0, 0, 1.01], [0, 1, 1, 0, 0, 0, float("nan")], [0] * 6]
)
def test_public_targets_reject_bad_bounds_shape_and_nonfinite(value: list[float]) -> None:
    with pytest.raises(ValueError):
        public_to_model(value)


def test_feedback_can_report_compliance_beyond_target_envelope() -> None:
    feedback = public_to_model([0, 1, 1, 0, 0, 0, 1.01], targets=False)
    assert feedback[-1] == pytest.approx(1.01 * GRIPPER_STROKE)


def test_descriptor_preserves_native_gains_and_both_initial_fingers(artifact: Path) -> None:
    loaded = load_yam_artifact(artifact)
    instance = loaded.robot_instance()
    assert tuple(j.name for j in instance.config.joints) == PUBLIC_JOINTS
    assert all(j.mode == "position" and j.kp == j.kd == 0 for j in instance.config.joints)
    assert instance.initial_positions[PUBLIC_JOINTS[-1]] == pytest.approx(0.01875 / GRIPPER_STROKE)
    assert instance.passive_joint_positions == {PASSIVE_JOINT: -0.01875}
    assert instance.config.joints[-1].ctrl_scale == GRIPPER_STROKE
    assert loaded.contract.dimos_revision == "fixture-not-a-revision"


def test_artifact_rejects_corrupt_xml_and_untrusted_manifest(artifact: Path) -> None:
    with pytest.raises(ValueError, match="contract SHA-256"):
        load_yam_artifact(artifact, expected_contract_sha256="0" * 64)
    with (artifact / "robot.xml").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch: robot.xml"):
        load_yam_artifact(artifact)


@pytest.mark.parametrize("mutation", ["outside", "order", "servo", "home", "scale"])
def test_artifact_rejects_inconsistent_physical_contract(artifact: Path, mutation: str) -> None:
    path = artifact / "contract.json"
    contract = json.loads(path.read_text())
    if mutation == "outside":
        contract["files"]["../outside"] = "0" * 64
    elif mutation == "order":
        contract["model_joint_names"].reverse()
    elif mutation == "servo":
        contract["actuators"][0]["gain"][0] += 1
    elif mutation == "home":
        contract["initial_positions"]["right_finger"] = 0
    else:
        contract["gripper_public_to_model_scale"] = 0.04
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        load_yam_artifact(artifact)


def test_unmanifested_mjcf_resource_rejected_before_compile(artifact: Path) -> None:
    path = artifact / "robot.xml"
    path.write_text(
        path.read_text().replace(
            "<worldbody>", '<asset><mesh name="external" file="unlisted.stl"/></asset><worldbody>'
        )
    )
    (artifact / "unlisted.stl").write_bytes(b"untrusted")
    contract = json.loads((artifact / "contract.json").read_text())
    contract["files"]["robot.xml"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (artifact / "contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="unmanifested"):
        load_yam_artifact(artifact)


def test_adapter_passes_the_policy_observation_without_rereading(mocker: MockerFixture) -> None:
    client = mocker.create_autospec(SimRobotClient, instance=True)
    client.descriptor = StreamDescriptor(
        "test", "yam", "test", 0.005, joint_names=PUBLIC_JOINTS, actuator_modes=("position",) * 7
    )
    observed = StateFrame(FrameMetadata(2, 3, 4, 0, 0.02), {})
    client.write_positions.return_value = 7
    adapter = YamSimAdapter(client)
    assert adapter.write_model_targets([0, 1, 1, 0, 0, 0, GRIPPER_STROKE], observed=observed) == 7
    client.write_positions.assert_called_once_with([0, 1, 1, 0, 0, 0, 1], observed=observed)
    client.read.assert_not_called()


@pytest.mark.native_e2e
@pytest.mark.timeout(30)
def test_native_loopback_client_command_reset_and_cleanup(artifact: Path) -> None:
    result = run_demo(artifact, steps=20)
    assert result["status"] == "completed"
    assert result["applied_command_sequences"] > 0
    assert result["max_native_coordinate_motion"] > 0.0001
    assert result["reset_passive_finger"] == -0.01875
    assert result["stale_observation_rejected"] is True
    assert result["physics_steps"] == 80
    # A second world can acquire and release its own real writer after cleanup.
    second = run_demo(artifact, steps=5)
    assert second["sim_id"] != result["sim_id"]
    assert second["applied_command_sequences"] > 0
