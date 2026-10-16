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

"""Map YAM leader readings onto dual OpenYAM follower commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_LEFT_ARM_JOINTS,
    DUAL_OPENYAM_RIGHT_ARM_JOINTS,
)

YamLeaderSide = Literal["left", "right"]
YAM_LEADER_SIDES: tuple[YamLeaderSide, ...] = ("left", "right")

# The uynitsuj YAM active leader: six arm servos then the gripper trigger, IDs 1-7.
# The calibration artifact names them joint_1..joint_7 (the OpenArm Mini schema).
YAM_LEADER_ARM_MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
YAM_LEADER_GRIPPER_MOTOR = "joint_7"
YAM_LEADER_MOTORS = (*YAM_LEADER_ARM_MOTORS, YAM_LEADER_GRIPPER_MOTOR)

# dual_openyam.urdf joint limits, identical for both arms. Sender-side clamp;
# the coordinator still enforces the hardware limits.
YAM_FOLLOWER_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.618, 3.1416),
    (0.0, 3.6652),
    (0.0, 3.1416),
    (-1.693, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0


def yam_follower_joint_names(side: YamLeaderSide) -> tuple[str, ...]:
    """Follower arm joints driven by a leader side, in leader motor order."""
    if side == "left":
        return tuple(DUAL_OPENYAM_LEFT_ARM_JOINTS)
    if side == "right":
        return tuple(DUAL_OPENYAM_RIGHT_ARM_JOINTS)
    raise ValueError(f"unknown YAM leader side {side!r}")


@dataclass(frozen=True)
class YamLeaderSideCommand:
    """Mapped command for one dual OpenYAM follower side."""

    side: YamLeaderSide
    positions_by_joint: dict[str, float]
    gripper_opening: float


def map_side_readings(
    side: YamLeaderSide,
    readings: Mapping[str, float],
    *,
    gripper_travel_radians: float,
    previous_positions_by_joint: Mapping[str, float] | None = None,
    max_joint_jump_radians: float | None = None,
) -> YamLeaderSideCommand:
    """Calibrated leader radians to follower joint positions and a gripper opening.

    Leader joint N drives follower joint N. The trigger reads 0 rad released;
    squeezing it through ``gripper_travel_radians`` closes the gripper.
    """
    missing = set(YAM_LEADER_MOTORS) - set(readings)
    if missing:
        raise ValueError(f"YAM leader readings missing motors: {sorted(missing)}")
    if gripper_travel_radians <= 0.0:
        raise ValueError("gripper_travel_radians must be positive")
    follower_joints = yam_follower_joint_names(side)
    positions_by_joint = {
        joint: _clamp(float(readings[motor]), *limits)
        for joint, motor, limits in zip(
            follower_joints, YAM_LEADER_ARM_MOTORS, YAM_FOLLOWER_JOINT_LIMITS, strict=True
        )
    }
    _validate_jump(positions_by_joint, previous_positions_by_joint, max_joint_jump_radians)
    squeeze = _clamp(
        abs(float(readings[YAM_LEADER_GRIPPER_MOTOR])) / gripper_travel_radians, 0.0, 1.0
    )
    return YamLeaderSideCommand(
        side=side,
        positions_by_joint=positions_by_joint,
        gripper_opening=GRIPPER_OPEN - squeeze * (GRIPPER_OPEN - GRIPPER_CLOSED),
    )


def combine_side_commands(commands: list[YamLeaderSideCommand]) -> JointState:
    """Combine side commands into one coordinator-facing JointState."""
    names: list[str] = []
    positions: list[float] = []
    for command in commands:
        names.extend(command.positions_by_joint)
        positions.extend(command.positions_by_joint.values())
    return JointState(name=names, position=positions)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _validate_jump(
    positions_by_joint: Mapping[str, float],
    previous_positions_by_joint: Mapping[str, float] | None,
    max_joint_jump_radians: float | None,
) -> None:
    if previous_positions_by_joint is None or max_joint_jump_radians is None:
        return
    for joint, position in positions_by_joint.items():
        previous = previous_positions_by_joint.get(joint)
        if previous is None:
            continue
        jump = abs(position - previous)
        if jump > max_joint_jump_radians:
            raise ValueError(
                f"YAM leader {joint} jump {jump:.3f} rad exceeds {max_joint_jump_radians:.3f} rad"
            )
