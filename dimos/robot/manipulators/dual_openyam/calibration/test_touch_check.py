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

from collections.abc import Callable
import threading
import time
from typing import Any

import numpy as np
import pinocchio as pin
import pytest

from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.dual_openyam.calibration.joint_stream import (
    JointStream,
    JointStreamError,
)
from dimos.robot.manipulators.dual_openyam.calibration.kinematics import (
    GRIPPER_HOUSING_FACE_Z,
    ArmKinematics,
    RigGeometry,
)
from dimos.robot.manipulators.dual_openyam.calibration.touch_check import (
    TouchResult,
    format_touch,
)
from dimos.robot.manipulators.dual_openyam.model import DUAL_OPENYAM_MODEL_PATH

LEFT = [f"left_joint{i}" for i in range(1, 7)]


def test_arm_chain_matches_the_urdf_grasp_frame() -> None:
    # Placed with the URDF's own spacing, the calibration chain must reproduce
    # the URDF's grasp frame exactly: same arm, only the base placement moves.
    kinematics = ArmKinematics()
    geometry = RigGeometry(base_spacing_m=0.62)
    model = pin.buildModelFromUrdf(str(DUAL_OPENYAM_MODEL_PATH))
    data = model.createData()
    rng = np.random.default_rng(7)
    for side in ("left", "right"):
        joints = rng.uniform(-0.5, 0.5, 6) + np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        q = pin.neutral(model)
        for index, value in enumerate(joints, start=1):
            q[model.joints[model.getJointId(f"{side}_joint{index}")].idx_q] = value
        pin.framesForwardKinematics(model, data, q)
        expected = data.oMf[model.getFrameId(f"{side}_grasp_frame")].translation
        actual = kinematics.point_in_rig(side, joints, np.array([0.0, 0.0, -0.1]), geometry)
        np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_fingertip_sits_finger_length_past_the_housing_face() -> None:
    geometry = RigGeometry(finger_length_m={"left": 0.08, "right": 0.09})
    assert geometry.fingertip_in_gripper("left")[2] == pytest.approx(GRIPPER_HOUSING_FACE_Z - 0.08)
    assert geometry.grasp_point_in_gripper("right")[2] == pytest.approx(
        GRIPPER_HOUSING_FACE_Z - 0.09 + 0.005
    )


def test_bases_sit_half_the_spacing_either_side() -> None:
    geometry = RigGeometry(base_spacing_m=0.43)
    np.testing.assert_allclose(geometry.base_position("left"), [0.0, 0.215, 0.0])
    np.testing.assert_allclose(geometry.base_position("right"), [0.0, -0.215, 0.0])


def test_touch_within_five_mm_passes() -> None:
    result = TouchResult(
        fingertips_m={
            "left": np.array([0.3, 0.0, -0.040]),
            "right": np.array([0.302, 0.003, -0.041]),
        },
        table_z_m=-0.042,
    )
    assert result.passed
    assert result.height_above_table_m("left") == pytest.approx(0.002)
    assert "PASSED" in format_touch(result)


def test_touch_sideways_mismatch_fails() -> None:
    result = TouchResult(
        fingertips_m={
            "left": np.array([0.3, 0.008, -0.042]),
            "right": np.array([0.3, 0.0, -0.042]),
        },
        table_z_m=-0.042,
    )
    assert not result.passed
    assert "FAILED" in format_touch(result)


class _FakeTransport:
    def __init__(self) -> None:
        self.callback: Callable[[JointState], Any] | None = None
        self.stopped = False

    def subscribe(self, callback: Callable[[JointState], Any]) -> Callable[[], None]:
        self.callback = callback
        return lambda: None

    def stop(self) -> None:
        self.stopped = True


def _publish(transport: _FakeTransport, values: Callable[[int], list[float]], count: int) -> None:
    def run() -> None:
        for index in range(count):
            assert transport.callback is not None
            transport.callback(JointState(name=LEFT, position=values(index)))
            time.sleep(0.01)

    threading.Thread(target=run, daemon=True).start()


def test_capture_averages_a_still_arm() -> None:
    transport = _FakeTransport()
    stream = JointStream(transport=transport)  # type: ignore[arg-type]
    _publish(transport, lambda i: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 + (i % 2) * 1e-4], 120)

    captured = stream.capture(LEFT, window_s=0.3, timeout_s=2.0)

    np.testing.assert_allclose(captured, [0.1, 0.2, 0.3, 0.4, 0.5, 0.60005], atol=1e-4)
    stream.close()
    assert transport.stopped


def test_capture_refuses_a_moving_arm() -> None:
    transport = _FakeTransport()
    stream = JointStream(transport=transport)  # type: ignore[arg-type]
    _publish(transport, lambda i: [0.01 * i, 0.2, 0.3, 0.4, 0.5, 0.6], 120)

    with pytest.raises(JointStreamError, match="moved"):
        stream.capture(LEFT, window_s=0.3, timeout_s=2.0)


def test_capture_without_a_publisher_says_so() -> None:
    stream = JointStream(transport=_FakeTransport())  # type: ignore[arg-type]

    with pytest.raises(JointStreamError, match="teleop session running"):
        stream.capture(LEFT, window_s=0.1, timeout_s=0.3)
