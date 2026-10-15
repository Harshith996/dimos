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

"""Touch grid: both arms touch known points on the table board.

The table board lies flat between the arms, long side left to right. Each arm
touches the four outer corners of its checker pattern, a 210 x 150 mm
rectangle, once with
the gripper straight down and once tilted. Per arm, a least-squares fit finds
the board's pose in that arm's base frame together with that arm's joint zero
errors; the tilted touches are what separate a joint zero from a base offset.
The two board poses then give the right base in the left base's frame: the
true spacing, height difference and turn between the arms.

Board frame: origin at the near-left pattern corner, x toward the far side, y toward
the left arm, z up out of the board. Near is the side closer to the arms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dimos.robot.manipulators.dual_openyam.calibration.kinematics import (
    ArmKinematics,
    RigGeometry,
)

TABLE_SQUARE_M = 0.030
# The four outer corners of the 7 x 5 table board's checker pattern, the
# corners of its black corner squares, in squares from near-left. They span
# 210 x 150 mm; the inner corners' 150 x 90 mm leaves joint 2 twice as uncertain.
_CORNER_SQUARES = {
    "near-left": (0, 0),
    "near-right": (0, -7),
    "far-left": (5, 0),
    "far-right": (5, -7),
}
GRID_POINT_NAMES = tuple(_CORNER_SQUARES)
DEFAULT_OFFSET_JOINTS = (2, 3)


def grid_points(square_m: float = TABLE_SQUARE_M) -> dict[str, np.ndarray]:
    return {
        name: np.array([far * square_m, left * square_m, 0.0])
        for name, (far, left) in _CORNER_SQUARES.items()
    }


def grid_path(session_dir: Path, side: str) -> Path:
    return session_dir / f"touch_grid_{side}.json"


def load_grid(session_dir: Path, side: str) -> list[dict[str, Any]]:
    path = grid_path(session_dir, side)
    return json.loads(path.read_text()) if path.is_file() else []


def save_grid(session_dir: Path, side: str, touches: Sequence[Mapping[str, Any]]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = grid_path(session_dir, side)
    path.write_text(json.dumps(list(touches), indent=2) + "\n")
    return path


def grid_touch(point: str, joints_rad: np.ndarray, taken_at: datetime) -> dict[str, Any]:
    return {
        "point": point,
        "taken_at": taken_at.isoformat(timespec="seconds"),
        "joints_rad": [float(v) for v in joints_rad],
    }


def finger_tilt_deg(kinematics: ArmKinematics, side: str, joints_rad: Sequence[float]) -> float:
    """Angle between the fingers and straight down, in the arm's base frame."""
    axis = kinematics.gripper_in_base(side, joints_rad).rotation @ np.array([0.0, 0.0, -1.0])
    return float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))


@dataclass(frozen=True)
class ArmGridFit:
    side: str
    board_in_base: pin.SE3
    offset_joints: tuple[int, ...]
    joint_offsets_rad: np.ndarray
    joint_offset_sigma_rad: np.ndarray
    board_translation_sigma_m: np.ndarray
    labels: tuple[str, ...]
    tilts_deg: tuple[float, ...]
    residuals_m: np.ndarray

    @property
    def rms_m(self) -> float:
        return float(np.sqrt(np.mean(np.sum(self.residuals_m**2, axis=1))))


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking ``source`` points onto ``target``."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, target_mean - rotation @ source_mean


def fit_arm(
    side: str,
    touches: Sequence[Mapping[str, Any]],
    kinematics: ArmKinematics,
    geometry: RigGeometry,
    *,
    square_m: float = TABLE_SQUARE_M,
    offset_joints: tuple[int, ...] = DEFAULT_OFFSET_JOINTS,
) -> ArmGridFit:
    """Board pose in the arm's base frame plus joint zero errors, by least squares.

    A joint offset is what to add to the encoder reading to get the true angle.
    """
    points = grid_points(square_m)
    labels = tuple(str(touch["point"]) for touch in touches)
    unknown = sorted(set(labels) - set(points))
    if unknown:
        raise ValueError(f"unknown grid points {unknown}; expected {list(points)}")
    if len(set(labels)) < 3:
        raise ValueError("touch at least three different corners")
    n_params = 6 + len(offset_joints)
    if 3 * len(touches) <= n_params:
        raise ValueError(f"need more than {n_params // 3} touches for {n_params} unknowns")

    joints = np.array([touch["joints_rad"] for touch in touches], dtype=np.float64)
    board_points = np.array([points[label] for label in labels])
    tip = geometry.fingertip_in_gripper(side)
    offset_index = [joint - 1 for joint in offset_joints]

    def tips(offsets: np.ndarray) -> np.ndarray:
        corrected = joints.copy()
        corrected[:, offset_index] += offsets
        return np.array([kinematics.gripper_in_base(side, q).act(tip) for q in corrected])

    rotation, translation = _kabsch(board_points, tips(np.zeros(len(offset_joints))))
    start = np.concatenate(
        [Rotation.from_matrix(rotation).as_rotvec(), translation, np.zeros(len(offset_joints))]
    )

    def residual(x: np.ndarray) -> np.ndarray:
        board_rotation = Rotation.from_rotvec(x[:3]).as_matrix()
        expected = board_points @ board_rotation.T + x[3:6]
        return (tips(x[6:]) - expected).ravel()

    solution = least_squares(residual, start, x_scale="jac")
    dof = solution.fun.size - n_params
    variance = float(solution.fun @ solution.fun) / dof
    covariance = np.linalg.pinv(solution.jac.T @ solution.jac) * variance
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return ArmGridFit(
        side=side,
        board_in_base=pin.SE3(Rotation.from_rotvec(solution.x[:3]).as_matrix(), solution.x[3:6]),
        offset_joints=offset_joints,
        joint_offsets_rad=solution.x[6:].copy(),
        joint_offset_sigma_rad=sigma[6:],
        board_translation_sigma_m=sigma[3:6],
        labels=labels,
        tilts_deg=tuple(finger_tilt_deg(kinematics, side, q) for q in joints),
        residuals_m=solution.fun.reshape(-1, 3),
    )


def right_base_in_left_base(left: ArmGridFit, right: ArmGridFit) -> pin.SE3:
    """Where the right arm's base really sits, in the left arm's base frame."""
    return left.board_in_base * right.board_in_base.inverse()


def format_arm_fit(fit: ArmGridFit) -> str:
    lines = [
        f"{fit.side} arm: {len(fit.labels)} touches, fit residual {fit.rms_m * 1000:.1f} mm rms"
    ]
    for joint, offset, sigma in zip(
        fit.offset_joints, fit.joint_offsets_rad, fit.joint_offset_sigma_rad, strict=True
    ):
        lines.append(
            f"  joint {joint} zero error {np.degrees(offset):+.2f} +/- {np.degrees(sigma):.2f} deg"
        )
    for label, tilt, residual in zip(fit.labels, fit.tilts_deg, fit.residuals_m, strict=True):
        lines.append(
            f"  {label:<11} tilt {tilt:4.0f} deg   miss {np.linalg.norm(residual) * 1000:4.1f} mm"
        )
    return "\n".join(lines)


def format_bases(relative: pin.SE3, geometry: RigGeometry) -> str:
    x, y, z = relative.translation * 1000.0
    yaw, pitch, roll = Rotation.from_matrix(relative.rotation).as_euler("ZYX", degrees=True)
    nominal = geometry.base_spacing_m * 1000.0
    return "\n".join(
        [
            "Right base seen from the left base:",
            f"  spacing   {-y:6.1f} mm   (tape {nominal:.0f} mm, {-y - nominal:+.1f})",
            f"  forward   {x:+6.1f} mm   (model 0)",
            f"  height    {z:+6.1f} mm   (model 0)",
            f"  turn      {yaw:+6.2f} deg  tilt {pitch:+.2f} / {roll:+.2f} deg   (model 0)",
        ]
    )
