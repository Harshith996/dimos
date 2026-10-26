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

from dataclasses import replace

import mujoco
import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from dimos.hardware.whole_body.spec import MotorCommand
from dimos.robot.manipulators.dual_openyam.spatial_model import KnownYamModel, write_ik_model
from dimos.robot.manipulators.dual_openyam.spatial_simulation import (
    DualYamSimAdapter,
    _station_robot_physics,
    spatial_world,
)
from dimos.robot.manipulators.dual_openyam.stations.anvil import ANVIL
from dimos.robot.manipulators.openyam.conftest import artifact as artifact
from dimos.robot.manipulators.openyam.simulation import (
    GRIPPER_STROKE,
    MODEL_JOINTS,
    load_yam_artifact,
)
from dimos.sim2.client import SimRobotClient
from dimos.sim2.messages import FrameMetadata, StateFrame, robot_descriptor
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.sensors.spec import Mount
from dimos.sim2.world.scene import load_scene


@pytest.mark.parametrize("angle", [-0.5, 0, 0.5])
def test_ik_frames_match_feedback_rendering_and_fingers_follow_opening(artifact, tmp_path, angle):
    loaded = load_yam_artifact(artifact)
    path = write_ik_model(loaded, tmp_path / "dual.urdf")
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    q = np.zeros(model.nq)
    positions = {}
    for i, name in enumerate(model.names[1:], 1):
        q[model.joints[i].idx_q] = angle
        positions[name] = angle
    positions.update({"left_arm/gripper": 0.2, "right_arm/gripper": 0.8})
    pin.framesForwardKinematics(model, data, q)
    renderer = KnownYamModel(loaded)
    arms = renderer.feedback(positions)
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    station = replace(
        ANVIL,
        scene=scene,
        wrists={side: Camera("wrist", Mount("link_6")) for side in ("left", "right")},
    )
    native = load_scene(spatial_world(loaded, station))
    native_data = mujoco.MjData(native)
    for side in ("left", "right"):
        for name in MODEL_JOINTS[:6]:
            native_data.joint(f"{side}/{name}").qpos[0] = angle
    mujoco.mj_forward(native, native_data)
    for side in ["left", "right"]:
        expected = data.oMf[model.getFrameId(f"{side}_grasp_frame")]
        np.testing.assert_allclose(arms[side]["grasp"], expected.translation, atol=1e-12)
        np.testing.assert_allclose(
            Rotation.from_quat(arms[side]["orientation"]).as_matrix(), expected.rotation, atol=1e-12
        )
        site = native_data.site(f"{side}/grasp_site")
        np.testing.assert_allclose(site.xpos, expected.translation, atol=1e-12)
        np.testing.assert_allclose(site.xmat.reshape(3, 3), expected.rotation, atol=1e-12)
    assert renderer.data["left"].joint("left_finger").qpos[0] == pytest.approx(0.2 * GRIPPER_STROKE)
    assert renderer.data["right"].joint("right_finger").qpos[0] == pytest.approx(
        -0.8 * GRIPPER_STROKE
    )
    assert model.nq == 12


def test_larger_displayed_opening_separates_the_fingers(artifact):
    renderer = KnownYamModel(load_yam_artifact(artifact))
    positions = {f"{side}_joint{i}": 0.0 for side in ("left", "right") for i in range(1, 7)}
    positions.update({"left_arm/gripper": 0.0, "right_arm/gripper": 1.0})
    arms = renderer.feedback(positions)
    closed = renderer.data["left"]
    opened = renderer.data["right"]
    closed_gap = np.linalg.norm(closed.body("left").xpos - closed.body("right").xpos)
    open_gap = np.linalg.norm(opened.body("left").xpos - opened.body("right").xpos)
    assert closed_gap == pytest.approx(0)
    assert open_gap == pytest.approx(0.071)
    assert arms["left"]["opening"] == 0
    assert arms["right"]["opening"] == 1


@pytest.fixture
def adapter(artifact, mocker):
    loaded = load_yam_artifact(artifact)
    clients = [mocker.create_autospec(SimRobotClient, instance=True) for _ in range(2)]
    for i, client in enumerate(clients):
        client.read.return_value = StateFrame(
            FrameMetadata(1, 0, 1, 0, 0.005),
            {
                "position": np.array([i * 0.2, 1, 1, 0, 0, 0, 0.2 + i * 0.5]),
                "velocity": np.zeros(7),
                "effort": np.zeros(7),
            },
        )
    mocker.patch(
        "dimos.robot.manipulators.dual_openyam.spatial_simulation.SimRobotClient",
        side_effect=clients,
    )
    descriptions = {
        side: robot_descriptor("test", side, "test", loaded.robot_instance(), 0.005).to_dict()
        for side in ["left", "right"]
    }
    device = DualYamSimAdapter(descriptions)
    try:
        device.connect()
        for client in clients:
            np.testing.assert_allclose(
                client.write_positions.call_args.args[0],
                client.read.return_value.values["position"],
            )
            client.write_positions.reset_mock()
        yield device, clients
    finally:
        device.disconnect()


def test_sim_adapter_preserves_arm_order_and_normalized_gripper_opening(adapter):
    device, clients = adapter
    state = device.read_motor_states()
    assert state[0].q == 0
    assert state[6].q == 0.2
    assert [m.q for m in state[-2:]] == pytest.approx([0.2, 0.7])
    commands = [MotorCommand(q=m.q, dq=0) for m in state]
    assert device.write_motor_commands(commands)
    for client in clients:
        np.testing.assert_allclose(
            client.write_positions.call_args.args[0], client.read.return_value.values["position"]
        )
        assert client.write_positions.call_args.kwargs["observed"] is client.read.return_value


def test_sim_reset_invalidates_connection_and_bad_targets_do_not_write(adapter):
    device, clients = adapter
    assert not device.write_motor_commands([MotorCommand(q=float("nan"), dq=0)] * 14)
    clients[0].write_positions.assert_not_called()
    clients[1].read.return_value = replace(
        clients[1].read.return_value, metadata=FrameMetadata(2, 1, 2, 0, 0.01)
    )
    assert not device.has_motor_states()
    assert not device.is_connected()
    assert not device.write_motor_commands([MotorCommand(q=0, dq=0)] * 14)
    for client in clients:
        client.write_positions.assert_not_called()


def test_station_compensates_robot_gravity_while_workspace_objects_still_fall():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <body name="left/link"><joint type="slide" axis="0 0 1"/>
            <geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
          <body name="payload"><joint type="slide" axis="0 0 1"/>
            <geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
        </worldbody></mujoco>
    """)
    _station_robot_physics(model, "left/")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, [0, -9.81], atol=1e-12)
