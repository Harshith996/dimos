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

import pytest

from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_LEFT_ARM_JOINTS,
    DUAL_OPENYAM_RIGHT_ARM_JOINTS,
)
from dimos.teleop.yam_leader.mapping import (
    YAM_FOLLOWER_JOINT_LIMITS,
    YAM_LEADER_MOTORS,
    combine_side_commands,
    map_side_readings,
    yam_follower_joint_names,
)


def _readings(**overrides: float) -> dict[str, float]:
    readings = {f"joint_{i}": 0.1 * i for i in range(1, 8)}
    readings.update(overrides)
    return readings


def test_leader_joint_n_drives_follower_joint_n_per_side() -> None:
    left = map_side_readings("left", _readings(), gripper_travel_radians=1.0)
    right = map_side_readings("right", _readings(), gripper_travel_radians=1.0)

    assert list(left.positions_by_joint) == DUAL_OPENYAM_LEFT_ARM_JOINTS
    assert list(right.positions_by_joint) == DUAL_OPENYAM_RIGHT_ARM_JOINTS
    assert list(left.positions_by_joint.values()) == pytest.approx([0.1 * i for i in range(1, 7)])
    assert yam_follower_joint_names("left") == tuple(DUAL_OPENYAM_LEFT_ARM_JOINTS)
    with pytest.raises(ValueError):
        yam_follower_joint_names("centre")  # type: ignore[arg-type]


def test_positions_clamp_to_the_yam_urdf_limits() -> None:
    command = map_side_readings(
        "left", _readings(joint_1=9.0, joint_2=-1.0, joint_6=-9.0), gripper_travel_radians=1.0
    )
    positions = list(command.positions_by_joint.values())

    assert positions[0] == YAM_FOLLOWER_JOINT_LIMITS[0][1]
    assert positions[1] == YAM_FOLLOWER_JOINT_LIMITS[1][0]
    assert positions[5] == YAM_FOLLOWER_JOINT_LIMITS[5][0]


def test_trigger_squeeze_closes_the_gripper() -> None:
    released = map_side_readings("left", _readings(joint_7=0.0), gripper_travel_radians=1.0)
    half = map_side_readings("left", _readings(joint_7=0.5), gripper_travel_radians=1.0)
    full = map_side_readings("left", _readings(joint_7=1.4), gripper_travel_radians=1.0)
    mirrored = map_side_readings("left", _readings(joint_7=-0.5), gripper_travel_radians=1.0)

    assert released.gripper_opening == 1.0
    assert half.gripper_opening == pytest.approx(0.5)
    assert full.gripper_opening == 0.0
    assert mirrored.gripper_opening == pytest.approx(0.5)
    with pytest.raises(ValueError):
        map_side_readings("left", _readings(), gripper_travel_radians=0.0)


def test_missing_motor_and_joint_jump_are_rejected() -> None:
    readings = _readings()
    del readings["joint_7"]
    with pytest.raises(ValueError, match="missing motors"):
        map_side_readings("left", readings, gripper_travel_radians=1.0)

    first = map_side_readings("left", _readings(), gripper_travel_radians=1.0)
    with pytest.raises(ValueError, match="jump"):
        map_side_readings(
            "left",
            _readings(joint_3=2.5),
            gripper_travel_radians=1.0,
            previous_positions_by_joint=first.positions_by_joint,
            max_joint_jump_radians=0.75,
        )
    assert len(YAM_LEADER_MOTORS) == 7


def test_combine_keeps_left_then_right_order() -> None:
    left = map_side_readings("left", _readings(), gripper_travel_radians=1.0)
    right = map_side_readings("right", _readings(joint_1=0.9), gripper_travel_radians=1.0)
    joint_state = combine_side_commands([left, right])

    assert list(joint_state.name) == [*DUAL_OPENYAM_LEFT_ARM_JOINTS, *DUAL_OPENYAM_RIGHT_ARM_JOINTS]
    assert joint_state.position[0] == pytest.approx(0.1)
    assert joint_state.position[6] == pytest.approx(0.9)
