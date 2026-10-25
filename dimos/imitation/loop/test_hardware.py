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

import numpy as np
import pytest
from reactivex.subject import Subject

from dimos.core.run_registry import RunEntry
from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.spec import MotorCommand, MotorState, WholeBodyAdapter
from dimos.imitation.loop.hardware import (
    OpenYamHardware,
    PhysicalYamDevice,
    StationConfig,
    station_owners,
)
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.robot.manipulators.openyam.simulation import YamArtifact


@pytest.fixture
def device(mocker):
    adapter = mocker.create_autospec(WholeBodyAdapter, instance=True)
    adapter.activate = mocker.Mock(return_value=True)
    adapter.read_motor_states.return_value = [MotorState(q=0.1)] * 6 + [MotorState(q=0.8)]
    adapter.write_motor_commands.return_value = True
    value = PhysicalYamDevice(
        adapter,
        JointLimits(
            position_lower=[-2.0] * 6 + [0.0],
            position_upper=[2.0] * 6 + [1.0],
            velocity_max=[0.5] * 7,
        ),
    )
    yield value
    value.disconnect()


def test_connection_does_not_enable_or_produce_training_feedback(device):
    device.connect()
    device.read_motor_states()
    device.adapter.activate.assert_not_called()
    assert device.snapshot() is None
    assert not device.write_motor_commands([MotorCommand(q=0.2)] * 7)


def test_driver_opening_is_converted_once_and_only_accepted_writes_are_recorded(device):
    assert device.activate()
    device.transition("human")
    feedback = device.read_motor_states()
    assert feedback[-1].q == pytest.approx(0.2)
    commands = [MotorCommand(q=0.1, kp=80, kd=5)] * 6 + [MotorCommand(q=0.7)]
    assert device.write_motor_commands(commands)
    dispatched = device.adapter.write_motor_commands.call_args.args[0]
    assert dispatched[:-1] == commands[:-1]
    assert dispatched[-1].q == pytest.approx(0.3)
    device.read_motor_states()
    accepted = device.snapshot()[2]
    assert accepted.positions[-1] == 0.7
    assert accepted.authority == "human"
    assert accepted.generation == device.observed_generation() == 1
    device.adapter.write_motor_commands.return_value = False
    assert not device.write_motor_commands([replace(command, q=0.4) for command in commands])
    device.read_motor_states()
    assert device.snapshot()[2] == accepted


def test_authority_change_invalidates_a_tick_that_read_before_takeover(device):
    device.activate()
    device.read_motor_states()
    device.transition("correction")
    assert not device.write_motor_commands([MotorCommand(q=0.3)] * 7)
    device.adapter.write_motor_commands.assert_not_called()
    device.read_motor_states()
    assert device.write_motor_commands([MotorCommand(q=0.3)] * 7)


def test_failed_activation_never_unlocks_commands(device):
    device.adapter.activate.return_value = False
    assert not device.activate()
    device.read_motor_states()
    assert not device.write_motor_commands([MotorCommand(q=0.3)] * 7)
    assert not device.enabled


@pytest.fixture
def hardware(mocker):
    artifact = mocker.create_autospec(YamArtifact, instance=True)
    artifact.contract_sha256 = "artifact"
    joint = mocker.Mock(lower=-2.0, upper=2.0, max_velocity=0.5)
    artifact.robot_instance.return_value.config.joints = [joint] * 7
    mocker.patch("dimos.imitation.loop.hardware.import_module")
    mocker.patch("dimos.imitation.loop.hardware.YamControl", autospec=True)
    mocker.patch("dimos.imitation.loop.hardware.station_owners", return_value=[])
    camera = mocker.patch("dimos.imitation.loop.hardware.Webcam", autospec=True)
    images = Subject()
    camera.return_value.image_stream.return_value = images
    station = StationConfig(
        name="right",
        can_device="right-can",
        camera_kind="webcam",
        camera_serial="avfoundation:camera",
    )
    robot = OpenYamHardware(artifact, station)
    yield robot, images
    robot.__exit__()
    images.dispose()


def test_usb_camera_flows_to_preview_and_teleop_without_enabling_motors(hardware, mocker):
    robot, images = hardware
    mocker.patch("dimos.imitation.loop.hardware.time.time", return_value=10.0)
    with robot:
        images.on_next(
            Image.from_numpy(np.full((4, 6, 3), 100, np.uint8), format=ImageFormat.RGB, ts=9.9)
        )
        assert robot.preflight()["camera_ready"]
        assert robot.preview.ts == 9.9
        assert robot.preview.data.shape == (180, 320, 3)
        assert np.all(robot.preview.data == 100)
        robot.control.publish_camera.assert_called_once_with(robot.preview)
        robot.device.adapter.activate.assert_not_called()
        robot.control.start.assert_called_once_with(enable=False)
    assert not images.observers


def test_camera_loss_revokes_preview_and_blocks_activation(hardware):
    robot, images = hardware
    with robot:
        images.on_error(RuntimeError("camera unplugged"))
        assert robot.preview is None
        assert robot.preflight()["camera_error"] == "camera unplugged"
        with pytest.raises(RuntimeError, match="wrist camera"):
            robot.enable()
        robot.device.adapter.activate.assert_not_called()


def test_busy_station_is_rejected_before_any_connection(hardware, mocker):
    robot, _ = hardware
    mocker.patch(
        "dimos.imitation.loop.hardware.station_owners",
        return_value=[{"blueprint": "teleop", "pid": 123}],
    )
    with pytest.raises(RuntimeError, match="Ctrl-C"):
        robot.__enter__()
    robot.control.start.assert_not_called()
    robot.device.adapter.connect.assert_not_called()


def test_ownership_matches_both_dual_arm_ports_but_not_an_unrelated_arm(mocker):
    run = RunEntry(
        "run",
        123,
        "teleop-webxr-dual-openyam",
        "now",
        "/tmp/log",
        original_argv=[
            "run",
            "teleop-webxr-dual-openyam",
            "--left-can-port",
            "left",
            "--right-can-port=right",
        ],
    )
    mocker.patch("dimos.imitation.loop.hardware.list_runs", return_value=[run])
    assert station_owners("left")[0]["pid"] == 123
    assert station_owners("right")[0]["pid"] == 123
    assert station_owners("other") == []


def test_camera_identity_changes_training_contract_but_input_hand_does_not():
    station = StationConfig(
        name="desk", can_device="serial", camera_serial="avfoundation:one", camera_kind="webcam"
    )
    assert (
        station.model_copy(update={"controller_hand": "left"}).observation_profile
        == station.observation_profile
    )
    assert (
        station.model_copy(update={"camera_serial": "avfoundation:two"}).observation_profile
        != station.observation_profile
    )
