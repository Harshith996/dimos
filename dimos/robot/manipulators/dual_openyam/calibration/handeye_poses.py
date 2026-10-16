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

"""Arm poses for the automatic hand-eye capture with the gripper board on the jig.

    python -m dimos.robot.manipulators.dual_openyam.calibration.handeye_poses --side right

Candidates come from a joint grid filtered by ``gravity_poses.ArmModel.pose_ok``
(table clearance, in front, on its own side, no joint overloaded) plus two
camera constraints evaluated with the provisional workcell calibration: the
board on the jig must face the overhead camera within ``max_facing_deg`` and
its centre must project well inside the image. The board plane is taken as
the jig's face: normal along the gripper +z by default (pointing back along
the fingers toward the wrist), centre at the closed fingertips; both are
parameters. Poses are then picked greedily for spread in joint space, each
reachable from the previous one by a table-clearing joint ramp.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import typer

from dimos.robot.manipulators.dual_openyam.calibration.board_vision import (
    GRIPPER_BOARD,
    Intrinsics,
    se3_from_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.gravity_poses import (
    ArmModel,
    Pose,
    check_side,
    home_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.kinematics import ArmKinematics

CALIBRATION_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = CALIBRATION_DIR / "dual_openyam_calibration_2026-10-15.json"
DEFAULT_POSE_COUNT = 20
MAX_FACING_DEG = 45.0
IMAGE_MARGIN_PX = 120
# The board's corners must also clear the table while the arm moves about.
BOARD_RADIUS_M = 0.09
MIN_BOARD_Z_ABOVE_TABLE_M = 0.05
# Degrees; wider than gravity_poses on joints 1 and 6 so the board turns and
# moves across the image, which is what pins down the camera pose.
HANDEYE_BOXES_DEG: dict[int, tuple[float, float, float]] = {
    1: (-40.0, 40.0, 10.0),
    2: (25.0, 150.0, 12.5),
    3: (25.0, 160.0, 13.5),
    4: (-75.0, 75.0, 37.5),
    5: (-75.0, 75.0, 37.5),
}
JOINT6_VALUES_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)

GRIPPER_AXES = {
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
}


@dataclass(frozen=True)
class JigBoard:
    """Where the board sits on the jig, in the gripper link frame (assumed, refined by the solve)."""

    normal_in_gripper: np.ndarray
    centre_in_gripper: np.ndarray

    @classmethod
    def from_options(
        cls, normal: str = "+z", centre_m: Sequence[float] = (0.0, 0.0, -0.146)
    ) -> JigBoard:
        if normal not in GRIPPER_AXES:
            raise ValueError(f"board normal must be one of {sorted(GRIPPER_AXES)}, got {normal!r}")
        return cls(GRIPPER_AXES[normal].copy(), np.asarray(centre_m, dtype=np.float64))

    def nominal_in_gripper(self) -> pin.SE3:
        """A board frame with z along the normal and the origin at the board's first corner."""
        z = self.normal_in_gripper / np.linalg.norm(self.normal_in_gripper)
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rotation = np.column_stack([x, y, z])
        half = np.array([GRIPPER_BOARD.size_m[0] / 2.0, GRIPPER_BOARD.size_m[1] / 2.0, 0.0])
        return pin.SE3(rotation, self.centre_in_gripper - rotation @ half)


@dataclass(frozen=True)
class CameraView:
    """The overhead camera as the provisional calibration places it in the rig."""

    camera_in_rig: pin.SE3
    intrinsics: Intrinsics
    base_in_rig: dict[str, pin.SE3]
    table_z_rig_m: float

    @classmethod
    def from_calibration(cls, calibration: Mapping[str, Any]) -> CameraView:
        return cls(
            camera_in_rig=se3_from_pose(calibration["overhead_camera"]["camera_in_rig"]),
            intrinsics=Intrinsics.from_calibration(calibration),
            base_in_rig={
                side: se3_from_pose(arm["base_in_rig"]) for side, arm in calibration["arms"].items()
            },
            table_z_rig_m=float(calibration["table"]["top_z_in_rig_m"]),
        )

    def gripper_in_rig(
        self, kinematics: ArmKinematics, side: str, joints_rad: Sequence[float]
    ) -> pin.SE3:
        return self.base_in_rig[side] * kinematics.gripper_in_base(side, list(joints_rad))

    def board_view(self, gripper_in_rig: pin.SE3, jig: JigBoard) -> tuple[float, np.ndarray, float]:
        """(facing angle deg, centre pixel, lowest board point above the table m)."""
        centre = gripper_in_rig.act(jig.centre_in_gripper)
        normal = gripper_in_rig.rotation @ jig.normal_in_gripper
        to_camera = self.camera_in_rig.translation - centre
        distance = float(np.linalg.norm(to_camera))
        cosine = float(normal @ to_camera) / max(distance, 1e-9)
        facing = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        in_camera = self.camera_in_rig.actInv(centre)
        pixel = (
            self.intrinsics.project(in_camera)[0]
            if in_camera[2] > 0.05
            else np.array([math.nan, math.nan])
        )
        # The board's rim, as a disk around the centre in the board plane.
        u = np.cross(normal, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        rim = centre + BOARD_RADIUS_M * (
            np.outer(np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)), u)
            + np.outer(np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False)), v)
        )
        lowest = float(rim[:, 2].min() - self.table_z_rig_m)
        return facing, pixel, lowest

    def board_visible(
        self,
        gripper_in_rig: pin.SE3,
        jig: JigBoard,
        *,
        max_facing_deg: float = MAX_FACING_DEG,
        margin_px: int = IMAGE_MARGIN_PX,
    ) -> bool:
        facing, pixel, lowest = self.board_view(gripper_in_rig, jig)
        if not np.isfinite(pixel).all() or facing > max_facing_deg:
            return False
        if lowest < MIN_BOARD_Z_ABOVE_TABLE_M:
            return False
        return bool(
            margin_px <= pixel[0] <= self.intrinsics.width - margin_px
            and margin_px <= pixel[1] <= self.intrinsics.height - margin_px
        )


@dataclass(frozen=True)
class HandEyePoseSet:
    poses: list[Pose]
    candidates: int
    facing_deg: tuple[float, ...]
    notes: list[str]
    side: str


def _grid(box: tuple[float, float, float]) -> np.ndarray:
    low, high, step = box
    return np.linspace(low, high, round((high - low) / step) + 1)


def candidate_handeye_poses(
    arm: ArmModel,
    view: CameraView,
    jig: JigBoard,
    kinematics: ArmKinematics,
    *,
    max_facing_deg: float = MAX_FACING_DEG,
) -> list[tuple[float, ...]]:
    """Grid poses (joints 1-5, joint 6 cycled) that clear the table and show the board to the camera."""
    grids = [_grid(HANDEYE_BOXES_DEG[j]) for j in range(1, 6)]
    found: list[tuple[float, ...]] = []
    j6_cycle = itertools.cycle(JOINT6_VALUES_DEG)
    for combo in itertools.product(*grids):
        joints = np.radians([*combo, 0.0])
        if not arm.pose_ok(joints):
            continue
        joints[5] = math.radians(next(j6_cycle))
        gripper = view.gripper_in_rig(kinematics, arm.side, joints)
        if view.board_visible(gripper, jig, max_facing_deg=max_facing_deg):
            found.append(tuple(float(v) for v in joints))
    return found


def select_spread(
    arm: ArmModel,
    candidates: Sequence[tuple[float, ...]],
    *,
    count: int,
    start: Sequence[float],
) -> list[tuple[float, ...]]:
    """Greedy farthest-point in joint space; each pick must be a table-clearing ramp from the last."""
    if not candidates:
        return []
    points = np.asarray(candidates)
    chosen: list[tuple[float, ...]] = []
    remaining = list(range(len(candidates)))
    last = np.asarray(start, dtype=np.float64)
    min_dist = np.full(len(candidates), np.inf)
    anchor = np.asarray(start)
    while len(chosen) < count and remaining:
        min_dist = np.minimum(min_dist, np.linalg.norm(points - anchor, axis=1))
        order = sorted(remaining, key=lambda i: -min_dist[i])
        picked = None
        for index in order:
            if arm.transition_ok(last, candidates[index]):
                picked = index
                break
        if picked is None:
            break
        chosen.append(candidates[picked])
        remaining.remove(picked)
        anchor = points[picked]
        last = points[picked]
    return chosen


def generate_handeye_poses(
    side: str,
    *,
    calibration: Mapping[str, Any],
    jig: JigBoard | None = None,
    count: int = DEFAULT_POSE_COUNT,
    max_facing_deg: float = MAX_FACING_DEG,
    arm: ArmModel | None = None,
    kinematics: ArmKinematics | None = None,
) -> HandEyePoseSet:
    side = check_side(side)
    arm = arm or ArmModel(side=side)
    kinematics = kinematics or ArmKinematics()
    jig = jig or JigBoard.from_options()
    view = CameraView.from_calibration(calibration)
    home = home_pose(side)
    candidates = candidate_handeye_poses(arm, view, jig, kinematics, max_facing_deg=max_facing_deg)
    notes: list[str] = []
    if not candidates:
        raise RuntimeError(
            "no pose shows the board to the camera; check the board normal/centre options"
        )
    selected = select_spread(arm, candidates, count=count, start=home.joints_rad)
    if len(selected) < count:
        notes.append(f"only {len(selected)} of {count} poses are reachable in sequence")
    if selected and not arm.transition_ok(selected[-1], home.joints_rad):
        notes.append("the last pose cannot ramp straight back home; return via the previous one")
    facing = tuple(
        view.board_view(view.gripper_in_rig(kinematics, side, p), jig)[0] for p in selected
    )
    return HandEyePoseSet(
        poses=[Pose(f"handeye_{i:02d}", p, side) for i, p in enumerate(selected, 1)],
        candidates=len(candidates),
        facing_deg=facing,
        notes=notes,
        side=side,
    )


def format_pose_set(pose_set: HandEyePoseSet) -> str:
    lines = [
        f"{pose_set.side} arm: {len(pose_set.poses)} hand-eye poses from {pose_set.candidates} "
        "candidates; joints 1..6 deg, board facing angle to the camera",
    ]
    for pose, facing in zip(pose_set.poses, pose_set.facing_deg, strict=True):
        joints = " ".join(f"{v:7.1f}" for v in pose.joints_deg)
        lines.append(f"{pose.name:12s} [{joints}]  facing {facing:5.1f} deg")
    lines.extend(f"note: {n}" for n in pose_set.notes)
    return "\n".join(lines)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    side: str = typer.Option("right", help="Which arm: left or right"),
    poses: int = typer.Option(DEFAULT_POSE_COUNT, help="How many poses"),
    board_normal: str = typer.Option(
        "+z", help="Board normal in the gripper frame: +z, -z, +x, ..."
    ),
    board_centre_m: str = typer.Option("0,0,-0.146", help="Board centre in the gripper frame, m"),
    calibration: Path = typer.Option(DEFAULT_CALIBRATION, help="Workcell calibration JSON"),
) -> None:
    """Print the hand-eye capture poses for one arm."""
    import json

    jig = JigBoard.from_options(board_normal, [float(v) for v in board_centre_m.split(",")])
    pose_set = generate_handeye_poses(
        side, calibration=json.loads(calibration.read_text()), jig=jig, count=poses
    )
    typer.echo(format_pose_set(pose_set))


if __name__ == "__main__":
    app()
