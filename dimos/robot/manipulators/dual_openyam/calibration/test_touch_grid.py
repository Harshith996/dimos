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

"""The touch-grid fit recovers known joint zero errors and base placement.

Touches are synthesised by inverse kinematics: for each corner and tilt, find
the true joint angles that put the fingertip on the corner, then report the
encoder reading the arm would show with the given zero errors.
"""

from typing import Any

import numpy as np
import pinocchio as pin
import pytest
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dimos.robot.manipulators.dual_openyam.calibration.kinematics import (
    ArmKinematics,
    RigGeometry,
)
from dimos.robot.manipulators.dual_openyam.calibration.touch_grid import (
    GRID_POINT_NAMES,
    fit_arm,
    grid_points,
    right_base_in_left_base,
)

GEOMETRY = RigGeometry()
# A touch posture captured on the rig (left arm on the centre cross), mirrored for the right.
_SEED_DEG = {
    "left": [-30.9, 131.4, 99.5, -57.8, -5.0, -38.6],
    "right": [31.0, 119.8, 85.0, -40.4, -7.1, 22.2],
}
# Straight down, then the fingertips tipped 20 deg forward and 20 deg away from
# the arm's own base: the wrist leans back toward the arm, which keeps the far
# corners in reach.
_TILTS = {
    "left": [(0.0, 0.0), (0.0, -20.0), (-20.0, 0.0)],
    "right": [(0.0, 0.0), (0.0, -20.0), (20.0, 0.0)],
}


@pytest.fixture(scope="module")
def kinematics() -> ArmKinematics:
    return ArmKinematics()


def _true_joints(
    kinematics: ArmKinematics, side: str, target: np.ndarray, finger_axis: np.ndarray
) -> np.ndarray:
    tip = GEOMETRY.fingertip_in_gripper(side)

    def residual(q: np.ndarray) -> np.ndarray:
        gripper = kinematics.gripper_in_base(side, q)
        axis = gripper.rotation @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([gripper.act(tip) - target, 0.05 * (axis - finger_axis)])

    solution = least_squares(
        residual, np.radians(_SEED_DEG[side]), xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    assert np.linalg.norm(solution.fun[:3]) < 1e-6
    return solution.x


def _touches(
    kinematics: ArmKinematics,
    side: str,
    board_in_base: pin.SE3,
    offsets_deg: dict[int, float],
    rng: np.random.Generator,
    noise_m: float = 0.0,
) -> list[dict[str, Any]]:
    touches = []
    for name, point in grid_points().items():
        for tilt_x, tilt_y in _TILTS[side]:
            axis = Rotation.from_euler("xy", [tilt_x, tilt_y], degrees=True).apply([0, 0, -1.0])
            target = board_in_base.act(point) + rng.normal(0.0, noise_m, 3)
            true = _true_joints(kinematics, side, target, axis)
            encoder = true.copy()
            for joint, offset in offsets_deg.items():
                encoder[joint - 1] -= np.radians(offset)
            touches.append({"point": name, "joints_rad": encoder.tolist()})
    return touches


def _board_in_base(side: str, spacing: float, height: float, turn_deg: float) -> pin.SE3:
    """Board centred between the arms, near edge 22 cm out, seen from one arm's base."""
    rig_board = pin.SE3(np.eye(3), np.array([0.22, 0.105, -0.028]))
    sign = 1.0 if side == "left" else -1.0
    rig_base = pin.SE3(
        Rotation.from_euler("z", turn_deg if side == "right" else 0.0, degrees=True).as_matrix(),
        np.array([0.0, sign * spacing / 2.0, height if side == "right" else 0.0]),
    )
    return rig_base.inverse() * rig_board


def test_recovers_joint_zero_errors(kinematics: ArmKinematics) -> None:
    rng = np.random.default_rng(1)
    board = _board_in_base("left", 0.43, 0.0, 0.0)
    touches = _touches(kinematics, "left", board, {2: 1.5, 3: -2.0}, rng)

    fit = fit_arm("left", touches, kinematics, GEOMETRY)

    np.testing.assert_allclose(np.degrees(fit.joint_offsets_rad), [1.5, -2.0], atol=0.02)
    np.testing.assert_allclose(fit.board_in_base.translation, board.translation, atol=2e-4)
    assert fit.rms_m < 1e-4


def test_recovers_where_the_right_base_sits(kinematics: ArmKinematics) -> None:
    rng = np.random.default_rng(2)
    spacing, height, turn = 0.418, 0.004, 1.5
    fits = {
        side: fit_arm(
            side,
            _touches(kinematics, side, _board_in_base(side, spacing, height, turn), {2: 0.8}, rng),
            kinematics,
            GEOMETRY,
        )
        for side in ("left", "right")
    }

    relative = right_base_in_left_base(fits["left"], fits["right"])

    np.testing.assert_allclose(relative.translation, [0.0, -spacing, height], atol=5e-4)
    yaw = Rotation.from_matrix(relative.rotation).as_euler("ZYX", degrees=True)[0]
    assert yaw == pytest.approx(turn, abs=0.05)


def test_touch_noise_shows_in_the_residual_and_sigma(kinematics: ArmKinematics) -> None:
    rng = np.random.default_rng(3)
    board = _board_in_base("left", 0.43, 0.0, 0.0)
    touches = _touches(kinematics, "left", board, {2: 1.0, 3: 1.0}, rng, noise_m=0.001)

    fit = fit_arm("left", touches, kinematics, GEOMETRY)

    assert 0.0003 < fit.rms_m < 0.002
    errors = np.abs(np.degrees(fit.joint_offsets_rad) - 1.0)
    assert np.all(errors < 4 * np.degrees(fit.joint_offset_sigma_rad) + 0.05)


def test_too_few_corners_is_refused(kinematics: ArmKinematics) -> None:
    touches = [{"point": GRID_POINT_NAMES[0], "joints_rad": [0.0] * 6}] * 4
    with pytest.raises(ValueError, match="three different corners"):
        fit_arm("left", touches, kinematics, GEOMETRY)
