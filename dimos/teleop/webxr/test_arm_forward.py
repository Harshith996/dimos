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

from collections.abc import Iterator
import json
import math
import time

import numpy as np
import pytest
from pytest_mock import MockerFixture

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Joy import Joy
from dimos.teleop.webxr.controller_types import Hand
from dimos.teleop.webxr.extensions import ArmTeleopModule


@pytest.fixture
def arm(mocker: MockerFixture) -> Iterator[ArmTeleopModule]:
    # Exercise real button edges without waiting for the debounce interval.
    mocker.patch("dimos.teleop.webxr.module.BUTTON_DEBOUNCE_S", 0.0)
    module = ArmTeleopModule()
    try:
        yield module
    finally:
        module.stop()


def _input(
    module: ArmTeleopModule,
    hand: Hand,
    orientation: Quaternion,
    position: Vector3 | None = None,
    *,
    grip: float = 0.0,
    click: bool = False,
) -> None:
    frame_id = hand.name.lower()
    pose = PoseStamped(
        position=position or Vector3(0.4, 1.1, -0.3),
        orientation=orientation,
        frame_id=frame_id,
        ts=123.0,
    )
    module._on_pose_bytes(pose.lcm_encode())
    joy = Joy(
        frame_id=frame_id,
        axes=[0.0, 0.0, 0.0, grip],
        buttons=[0, 0, 0, int(click), 0, 0, 0],
    )
    assert module._on_joy_bytes(joy.lcm_encode())


def _buttons(module: ArmTeleopModule) -> None:
    for _ in range(2):
        module._publish_button_state(
            module._controllers[Hand.LEFT], module._controllers[Hand.RIGHT]
        )


def _output(module: ArmTeleopModule, hand: Hand) -> PoseStamped:
    pose = module._get_output_pose(hand)
    assert pose is not None
    return pose


@pytest.mark.parametrize("click_hand", list(Hand))
@pytest.mark.parametrize("yaw", [-math.pi / 2, math.pi / 2, math.pi])
def test_calibration_maps_both_hands_translation_and_rotation_to_shared_heading(
    arm: ArmTeleopModule, click_hand: Hand, yaw: float
) -> None:
    heading = Quaternion.from_euler(Vector3(0.0, yaw, 0.0))
    tilt = Quaternion.from_euler(Vector3(0.3, 0.0, 0.0)) * Quaternion.from_euler(
        Vector3(0.0, 0.0, 0.2)
    )
    start = Vector3(0.4, 1.1, -0.3)
    for hand in Hand:
        _input(arm, hand, heading * tilt, start, click=hand == click_hand)
    _buttons(arm)
    before = {hand: _output(arm, hand) for hand in Hand}

    # Forward, right and up in the newly chosen operator frame.
    moved = start + heading.rotate_vector(Vector3(0.1, 0.3, -0.2))
    twist = Quaternion.from_euler(Vector3(0.0, 0.0, -0.4))
    for hand in Hand:
        _input(arm, hand, heading * twist * tilt, moved, click=hand == click_hand)
    _buttons(arm)  # Holding the stick must not continuously recalibrate.

    for hand in Hand:
        after = _output(arm, hand)
        np.testing.assert_allclose(
            (after.position - before[hand].position).to_numpy(), [0.2, -0.1, 0.3], atol=1e-10
        )
        delta = after.orientation * before[hand].orientation.inverse()
        expected = Quaternion.from_euler(Vector3(0.4, 0.0, 0.0))
        np.testing.assert_allclose(
            delta.to_rotation_matrix(), expected.to_rotation_matrix(), atol=1e-10
        )
        assert after.ts == 123.0
        assert after.frame_id == hand.name.lower()


@pytest.mark.parametrize("grip_hand", list(Hand))
def test_either_grip_blocks_calibration_and_requires_a_new_click(
    arm: ArmTeleopModule, mocker: MockerFixture, grip_hand: Hand
) -> None:
    send = mocker.patch.object(arm, "_broadcast_text")
    heading = Quaternion.from_euler(Vector3(0.0, math.pi / 2, 0.0))
    for hand in Hand:
        _input(arm, hand, heading, grip=0.6 if hand == grip_hand else 0.0, click=hand == Hand.RIGHT)
    original = _output(arm, Hand.RIGHT)
    _buttons(arm)
    np.testing.assert_allclose(
        _output(arm, Hand.RIGHT).orientation.to_numpy(), original.orientation.to_numpy()
    )
    assert json.loads(send.call_args.args[0])["message"].startswith("Release both grips")

    for hand in Hand:
        _input(arm, hand, heading, click=hand == Hand.RIGHT)
    _buttons(arm)
    assert send.call_count == 1  # Releasing a grip while holding the stick does not calibrate.

    _input(arm, Hand.RIGHT, heading)
    _buttons(arm)
    _input(arm, Hand.RIGHT, heading, click=True)
    _buttons(arm)
    forward = _output(arm, Hand.RIGHT).orientation.rotate_vector(Vector3(0.0, 0.0, -1.0))
    np.testing.assert_allclose(forward.to_numpy(), [1.0, 0.0, 0.0], atol=1e-10)


def test_repeated_calibration_replaces_heading_and_disconnect_clears_it(
    arm: ArmTeleopModule,
) -> None:
    heading = Quaternion.from_euler(Vector3(0.0, math.pi / 2, 0.0))
    _input(arm, Hand.RIGHT, heading)
    original = _output(arm, Hand.RIGHT)
    _input(arm, Hand.RIGHT, heading, click=True)
    _buttons(arm)
    calibrated = _output(arm, Hand.RIGHT)
    _input(arm, Hand.RIGHT, heading)
    _buttons(arm)
    _input(arm, Hand.RIGHT, heading, click=True)
    _buttons(arm)
    np.testing.assert_allclose(
        _output(arm, Hand.RIGHT).orientation.to_numpy(), calibrated.orientation.to_numpy()
    )

    arm._reset_controller_state()
    _input(arm, Hand.RIGHT, heading)
    np.testing.assert_allclose(
        _output(arm, Hand.RIGHT).orientation.to_numpy(), original.orientation.to_numpy()
    )


@pytest.mark.parametrize("invalid", ["vertical", "stale", "missing"])
def test_invalid_pointing_pose_preserves_previous_heading(
    arm: ArmTeleopModule, mocker: MockerFixture, invalid: str
) -> None:
    send = mocker.patch.object(arm, "_broadcast_text")
    heading = Quaternion.from_euler(Vector3(0.0, math.pi / 2, 0.0))
    _input(arm, Hand.LEFT, heading, click=True)
    _buttons(arm)
    _input(arm, Hand.LEFT, heading)
    _buttons(arm)
    original = _output(arm, Hand.LEFT)

    _input(arm, Hand.RIGHT, Quaternion.from_euler(Vector3(math.pi / 2, 0.0, 0.0)), click=True)
    if invalid == "stale":
        arm._last_pose_update[Hand.RIGHT] = time.monotonic() - arm.config.input_timeout_s - 1.0
    elif invalid == "missing":
        arm._current_poses[Hand.RIGHT] = None
    _buttons(arm)

    np.testing.assert_allclose(
        _output(arm, Hand.LEFT).orientation.to_numpy(), original.orientation.to_numpy()
    )
    expected = (
        "Point the controller ahead" if invalid == "vertical" else "Controller tracking unavailable"
    )
    assert json.loads(send.call_args.args[0])["message"].startswith(expected)
