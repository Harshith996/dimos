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

"""Operator arm postures derived from WebXR body tracking."""

from collections.abc import Iterator

import pytest

from dimos.teleop.webxr.body_tracking import ArmPosture, BodyJointPose, BodyTrackingSnapshot
from dimos.teleop.webxr.controller_types import Hand
from dimos.teleop.webxr.extensions import ArmTeleopModule, arm_posture_from_body

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _snapshot(joints: dict[str, tuple[float, float, float]] | None) -> BodyTrackingSnapshot:
    return BodyTrackingSnapshot(
        type="body_tracking_snapshot",
        capture_time_s=5.0,
        frame_id="local-floor",
        joints=None
        if joints is None
        else {
            name: BodyJointPose(position=position, orientation=IDENTITY)
            for name, position in joints.items()
        },
    )


def test_arm_posture_maps_shoulder_elbow_and_wrist_into_the_robot_frame() -> None:
    snapshot = _snapshot(
        {
            "left-arm-upper": (1.0, 2.0, 3.0),
            "left-arm-lower": (0.1, 1.5, -0.4),
            "left-hand-wrist": (0.3, 1.2, -0.6),
            "left-hand-wrist-twist": (9.0, 9.0, 9.0),
        }
    )

    posture = arm_posture_from_body(snapshot, Hand.LEFT)

    # WebXR (x right, y up, z back) -> robot (x forward = -z, y left = -x, z up = y).
    assert posture == ArmPosture(
        ts=5.0,
        hand="left",
        wrist=(0.6, -0.3, 1.2),
        elbow=(0.4, -0.1, 1.5),
        shoulder=(-3.0, -1.0, 2.0),
    )


def test_arm_posture_falls_back_to_the_forearm_twist_joint() -> None:
    snapshot = _snapshot(
        {"right-arm-lower": (0.0, 1.0, 0.0), "right-hand-wrist-twist": (0.0, 1.0, -0.25)}
    )

    posture = arm_posture_from_body(snapshot, Hand.RIGHT)

    assert posture is not None
    assert posture.hand == "right"
    assert posture.wrist == pytest.approx((0.25, 0.0, 1.0))
    assert posture.shoulder is None


@pytest.mark.parametrize(
    "joints",
    [
        None,
        {},
        {"left-arm-lower": (0.0, 1.0, 0.0)},
        {"left-hand-wrist": (0.0, 1.0, 0.0)},
        {"right-arm-lower": (0.0, 1.0, 0.0), "right-hand-wrist": (0.0, 1.0, -0.3)},
    ],
)
def test_arm_posture_requires_that_arms_elbow_and_wrist(
    joints: dict[str, tuple[float, float, float]] | None,
) -> None:
    assert arm_posture_from_body(_snapshot(joints), Hand.LEFT) is None


@pytest.fixture
def module() -> Iterator[ArmTeleopModule]:
    module = ArmTeleopModule(server_port=9443, body_tracking_mode="optional")
    try:
        yield module
    finally:
        module.stop()


def test_body_snapshots_publish_a_posture_per_tracked_arm(module: ArmTeleopModule) -> None:
    left: list[ArmPosture] = []
    right: list[ArmPosture] = []
    module.left_arm_posture.subscribe(left.append)
    module.right_arm_posture.subscribe(right.append)
    snapshot = _snapshot({"left-arm-lower": (0.0, 1.3, -0.3), "left-hand-wrist": (0.0, 1.0, -0.5)})

    assert module._dispatch_text_message(snapshot.model_dump_json())
    assert module._dispatch_text_message(_snapshot(None).model_dump_json())

    assert [posture.elbow for posture in left] == [(0.3, 0.0, 1.3)]
    assert right == []
