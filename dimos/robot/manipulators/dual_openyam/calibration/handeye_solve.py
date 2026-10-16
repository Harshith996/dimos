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

"""Hand-eye solve for one arm from gripper-board captures (eye-to-hand).

    python -m dimos.robot.manipulators.dual_openyam.calibration.handeye_solve \\
        <captures.json | recordings/<run-id>> --side right \\
        [--touch-grid touch_grid_right.json --touch-frames <overhead PNG dir>]

Unknowns: the camera pose in the arm's base frame, the board pose on the
gripper, and the zero offsets of joints 2..5 (what to add to the encoder
reading). Residuals are the pixel errors of every detected board corner,
with a robust (soft L1) loss; poses whose own residual stands out are flagged
as outliers and the fit is repeated without them. Standard errors come from
the Jacobian at the solution.

Held-out check: table-board touches from the touch grid (``arm_checks grid``),
which this fit never sees. With the table board's pose from overhead frames,
the tool point (pin tip by default) each touch posture predicts is compared
with the touched corner, in millimetres.

Output: ``handeye_<side>_<date>.json`` next to the captures (gravity-style:
created, side, source, results) and ``handeye_report_<side>.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import typer

from dimos.robot.manipulators.dual_openyam.calibration.board_vision import (
    GRIPPER_BOARD,
    TABLE_BOARD,
    BoardDetection,
    BoardSpec,
    Intrinsics,
    SavedFrames,
    pose_to_dict,
    se3_from_pose,
    sight_board,
    solve_board_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.encoders import SIDES
from dimos.robot.manipulators.dual_openyam.calibration.kinematics import ArmKinematics
from dimos.robot.manipulators.dual_openyam.calibration.touch_grid import grid_points

CALIBRATION_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = CALIBRATION_DIR / "dual_openyam_calibration_2026-10-15.json"
DEFAULT_OFFSET_JOINTS = (2, 3, 4, 5)
MIN_RECORDS = 5
OUTLIER_FACTOR = 3.0
OUTLIER_FLOOR_PX = 1.5
CAPTURE_LABEL = "handeye_capture"
GRAVITY_EVENTS_STREAM = "gravity_events"
# The touch grid's board frame (origin near-left corner, x far, y left, z up;
# title at the far edge) expressed in the OpenCV board frame of the table board.
TOUCH_GRID_IN_BOARD = pin.SE3(
    np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), np.array([0.15, 0.0, 0.0])
).inverse()
FINGERTIP_IN_GRIPPER_M = np.array([0.0, 0.0, -0.146])
PIN_TIP_IN_GRIPPER_M = np.array([0.0, 0.0, -0.156])


@dataclass
class HandEyeRecord:
    """One still pose: encoder joints and the gripper board as the camera saw it."""

    joints_rad: list[float]
    detection: BoardDetection
    name: str = ""
    taken_at: str = ""
    rms_px: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "taken_at": self.taken_at,
            "joints_rad": [float(v) for v in self.joints_rad],
            "rms_px": self.rms_px,
            **self.detection.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], board: BoardSpec = GRIPPER_BOARD) -> HandEyeRecord:
        return cls(
            joints_rad=[float(v) for v in data["joints_rad"]],
            detection=BoardDetection.from_json(data, board),
            name=str(data.get("name", "")),
            taken_at=str(data.get("taken_at", "")),
            rms_px=data.get("rms_px"),
        )


def save_records(path: Path, records: Sequence[HandEyeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_json() for r in records], indent=1) + "\n")


def load_records(source: Path, side: str | None = None) -> list[HandEyeRecord]:
    """Records from a captures JSON, or from a recording's ``gravity_events`` markers."""
    source = Path(source)
    if source.is_dir():
        return records_from_recording(source, side)
    data = json.loads(source.read_text())
    if isinstance(data, dict):
        data = data["records"]
    return [HandEyeRecord.from_json(r) for r in data]


def records_from_recording(recording_dir: Path, side: str | None = None) -> list[HandEyeRecord]:
    from dimos.memory.store.sqlite import SqliteStore

    path = recording_dir / "memory.db"
    if not path.exists():
        raise FileNotFoundError(f"no memory.db in {recording_dir}")
    store = SqliteStore(path=str(path), must_exist=True)
    store.start()
    records: list[HandEyeRecord] = []
    try:
        if GRAVITY_EVENTS_STREAM not in set(store.list_streams()):
            raise ValueError(f"{path} has no {GRAVITY_EVENTS_STREAM} stream")
        for obs in store.stream(GRAVITY_EVENTS_STREAM):
            event = json.loads(obs.data.data)
            data = event.get("data", {})
            if not str(event.get("label", "")).startswith(CAPTURE_LABEL):
                continue
            if side is not None and data.get("side") != side:
                continue
            records.append(HandEyeRecord.from_json(data["record"]))
    finally:
        store.stop()
    return records


def _se3(rotvec: np.ndarray, translation: np.ndarray) -> pin.SE3:
    return pin.SE3(Rotation.from_rotvec(rotvec).as_matrix(), np.asarray(translation, dtype=float))


def _pack(pose: pin.SE3) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(pose.rotation).as_rotvec(), pose.translation])


@dataclass
class HandEyeFit:
    side: str
    camera_in_base: pin.SE3
    board_in_gripper: pin.SE3
    offset_joints: tuple[int, ...]
    offsets_rad: np.ndarray
    offset_sigma_rad: np.ndarray
    camera_sigma: np.ndarray  # rotvec (3) then translation (3)
    rms_px: float
    per_pose_rms_px: list[float]
    used: list[int]
    outliers: list[int]
    records: int

    def to_json(self) -> dict[str, Any]:
        return {
            "camera_in_base": pose_to_dict(self.camera_in_base),
            "board_in_gripper": pose_to_dict(self.board_in_gripper),
            "joint_zero_offsets_rad": {
                f"joint{j}": round(float(o), 6)
                for j, o in zip(self.offset_joints, self.offsets_rad, strict=True)
            },
            "joint_zero_offsets_sigma_deg": {
                f"joint{j}": round(float(math.degrees(s)), 3)
                for j, s in zip(self.offset_joints, self.offset_sigma_rad, strict=True)
            },
            "camera_sigma_rot_deg": [
                round(float(math.degrees(s)), 3) for s in self.camera_sigma[:3]
            ],
            "camera_sigma_mm": [round(float(s) * 1000, 2) for s in self.camera_sigma[3:]],
            "rms_px": round(self.rms_px, 3),
            "per_pose_rms_px": [round(v, 3) for v in self.per_pose_rms_px],
            "poses_used": self.used,
            "outlier_poses": self.outliers,
            "records": self.records,
        }


def _gripper(
    kinematics: ArmKinematics,
    side: str,
    joints: Sequence[float],
    index: list[int],
    offsets: np.ndarray,
) -> pin.SE3:
    q = np.array(joints, dtype=np.float64)
    q[index] += offsets
    return kinematics.gripper_in_base(side, list(q))


def initial_guess(
    side: str, records: Sequence[HandEyeRecord], kinematics: ArmKinematics, intrinsics: Intrinsics
) -> tuple[pin.SE3, pin.SE3]:
    """Park's closed form (eye-to-hand): camera in base, board in gripper."""
    cam_board = [solve_board_pose(r.detection, intrinsics).board_in_camera for r in records]
    base_gripper = [kinematics.gripper_in_base(side, r.joints_rad) for r in records]
    inv = [T.inverse() for T in base_gripper]
    rotation, translation = cv2.calibrateHandEye(
        [T.rotation for T in inv],
        [T.translation for T in inv],
        [T.rotation for T in cam_board],
        [T.translation for T in cam_board],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    base_in_camera = pin.SE3(rotation, np.asarray(translation).ravel())
    camera_in_base = base_in_camera.inverse()
    boards = [
        bg.inverse() * camera_in_base * cb for bg, cb in zip(base_gripper, cam_board, strict=True)
    ]
    # Average the per-pose board placements (translation mean, rotation of the median pose).
    translation_mean = np.mean([b.translation for b in boards], axis=0)
    rotvecs = np.array([Rotation.from_matrix(b.rotation).as_rotvec() for b in boards])
    board_in_gripper = _se3(Rotation.from_rotvec(rotvecs).mean().as_rotvec(), translation_mean)
    return camera_in_base, board_in_gripper


def fit_handeye(
    side: str,
    records: Sequence[HandEyeRecord],
    *,
    kinematics: ArmKinematics,
    intrinsics: Intrinsics,
    offset_joints: tuple[int, ...] = DEFAULT_OFFSET_JOINTS,
    f_scale_px: float = 1.0,
    reject_outliers: bool = True,
) -> HandEyeFit:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if len(records) < MIN_RECORDS:
        raise ValueError(f"need at least {MIN_RECORDS} poses, got {len(records)}")
    index = [j - 1 for j in offset_joints]
    camera0, board0 = initial_guess(side, records, kinematics, intrinsics)
    x0 = np.concatenate([_pack(camera0), _pack(board0), np.zeros(len(offset_joints))])
    sizes = [2 * r.detection.count for r in records]

    def residual_for(used: Sequence[int]):
        def residual(x: np.ndarray) -> np.ndarray:
            base_in_camera = _se3(x[:3], x[3:6]).inverse()
            board_in_gripper = _se3(x[6:9], x[9:12])
            out = []
            for i in used:
                r = records[i]
                T = (
                    base_in_camera
                    * _gripper(kinematics, side, r.joints_rad, index, x[12:])
                    * board_in_gripper
                )
                points = r.detection.object_points @ T.rotation.T + T.translation
                out.append((intrinsics.project(points) - r.detection.image_points_ideal).ravel())
            return np.concatenate(out)

        return residual

    def solve(used: list[int], start: np.ndarray):
        solution = least_squares(
            residual_for(used), start, loss="soft_l1", f_scale=f_scale_px, x_scale="jac"
        )
        chunks = np.split(solution.fun, np.cumsum([sizes[i] for i in used])[:-1])
        per_pose = {i: float(np.sqrt(np.mean(c**2))) for i, c in zip(used, chunks, strict=True)}
        return solution, per_pose

    used = list(range(len(records)))
    solution, per_pose = solve(used, x0)
    outliers: list[int] = []
    if reject_outliers and len(used) > MIN_RECORDS:
        median = float(np.median(list(per_pose.values())))
        threshold = max(OUTLIER_FACTOR * median, OUTLIER_FLOOR_PX)
        outliers = sorted(i for i, v in per_pose.items() if v > threshold)
        if outliers and len(used) - len(outliers) >= MIN_RECORDS:
            used = [i for i in used if i not in outliers]
            solution, per_pose = solve(used, solution.x)
    dof = max(1, solution.fun.size - x0.size)
    variance = float(solution.fun @ solution.fun) / dof
    covariance = np.linalg.pinv(solution.jac.T @ solution.jac) * variance
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    # Residuals of every record at the solution, outliers included, for the report.
    all_res = residual_for(list(range(len(records))))(solution.x)
    chunks = np.split(all_res, np.cumsum(sizes)[:-1])
    return HandEyeFit(
        side=side,
        camera_in_base=_se3(solution.x[:3], solution.x[3:6]),
        board_in_gripper=_se3(solution.x[6:9], solution.x[9:12]),
        offset_joints=tuple(offset_joints),
        offsets_rad=solution.x[12:].copy(),
        offset_sigma_rad=sigma[12:],
        camera_sigma=sigma[:6],
        rms_px=float(np.sqrt(np.mean(solution.fun**2))),
        per_pose_rms_px=[float(np.sqrt(np.mean(c**2))) for c in chunks],
        used=used,
        outliers=outliers,
        records=len(records),
    )


# Held-out check against the touch grid


@dataclass
class TouchCheck:
    tool_in_gripper_m: list[float]
    touches: int
    rms_mm: float
    worst_mm: float
    mean_offset_mm: list[float]  # camera frame
    per_touch_mm: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_in_gripper_m": self.tool_in_gripper_m,
            "touches": self.touches,
            "rms_mm": round(self.rms_mm, 2),
            "worst_mm": round(self.worst_mm, 2),
            "mean_offset_camera_mm": [round(v, 2) for v in self.mean_offset_mm],
            "per_touch_mm": self.per_touch_mm,
        }


def touch_check(
    fit: HandEyeFit,
    touches: Sequence[Mapping[str, Any]],
    table_board_in_camera: pin.SE3,
    *,
    kinematics: ArmKinematics,
    tool_in_gripper: np.ndarray = PIN_TIP_IN_GRIPPER_M,
) -> TouchCheck:
    """Tool point predicted from each touch posture against the touched corner, camera frame, mm."""
    index = [j - 1 for j in fit.offset_joints]
    corners = grid_points()
    base_in_camera = fit.camera_in_base.inverse()
    grid_in_camera = table_board_in_camera * TOUCH_GRID_IN_BOARD
    rows = []
    errors = []
    for touch in touches:
        point = str(touch["point"])
        if point not in corners:
            continue
        gripper = _gripper(kinematics, fit.side, touch["joints_rad"], index, fit.offsets_rad)
        predicted = (base_in_camera * gripper).act(np.asarray(tool_in_gripper))
        error = (predicted - grid_in_camera.act(corners[point])) * 1000.0
        errors.append(error)
        rows.append(
            {
                "point": point,
                "error_mm": [round(float(v), 2) for v in error],
                "norm_mm": round(float(np.linalg.norm(error)), 2),
            }
        )
    if not errors:
        raise ValueError("no touches at known grid points")
    e = np.array(errors)
    norms = np.linalg.norm(e, axis=1)
    return TouchCheck(
        tool_in_gripper_m=[float(v) for v in tool_in_gripper],
        touches=len(e),
        rms_mm=float(np.sqrt(np.mean(norms**2))),
        worst_mm=float(norms.max()),
        mean_offset_mm=[float(v) for v in e.mean(axis=0)],
        per_touch_mm=rows,
    )


# Report


def format_fit(fit: HandEyeFit, calibration: Mapping[str, Any] | None = None) -> str:
    cam = fit.camera_in_base
    lines = [
        f"{fit.side} arm: {fit.records} poses, {len(fit.used)} used, {fit.rms_px:.2f} px rms"
        + (f"; outliers {[i + 1 for i in fit.outliers]}" if fit.outliers else ""),
        f"camera in {fit.side} base: xyz {np.round(cam.translation * 1000, 1).tolist()} mm, "
        f"rpy {np.round(np.degrees(pin.rpy.matrixToRpy(cam.rotation)), 2).tolist()} deg "
        f"(+/- {np.round(fit.camera_sigma[3:] * 1000, 1).tolist()} mm, "
        f"{np.round(np.degrees(fit.camera_sigma[:3]), 2).tolist()} deg)",
        f"board in gripper: xyz {np.round(fit.board_in_gripper.translation * 1000, 1).tolist()} mm, "
        f"normal (board z in gripper) {np.round(fit.board_in_gripper.rotation[:, 2], 3).tolist()}",
        "joint zero offsets (add to the encoder reading): "
        + ", ".join(
            f"j{j} {math.degrees(o):+.2f} +/- {math.degrees(s):.2f} deg"
            for j, o, s in zip(
                fit.offset_joints, fit.offsets_rad, fit.offset_sigma_rad, strict=True
            )
        ),
        "per-pose rms px: " + " ".join(f"{v:.2f}" for v in fit.per_pose_rms_px),
    ]
    if calibration is not None:
        base_in_rig = se3_from_pose(calibration["arms"][fit.side]["base_in_rig"])
        camera_in_rig = base_in_rig * cam
        previous = se3_from_pose(calibration["overhead_camera"]["camera_in_rig"])
        delta = previous.inverse() * camera_in_rig
        lines.append(
            f"camera in rig: xyz {np.round(camera_in_rig.translation * 1000, 1).tolist()} mm, "
            f"rpy {np.round(np.degrees(pin.rpy.matrixToRpy(camera_in_rig.rotation)), 2).tolist()} deg; "
            f"moved {np.linalg.norm(delta.translation) * 1000:.1f} mm, "
            f"{math.degrees(np.linalg.norm(pin.log3(delta.rotation))):.2f} deg from the "
            f"{calibration.get('date', '')} calibration"
        )
        old = calibration["arms"][fit.side].get("joint_zero_offsets_rad", {})
        for j, _o in zip(fit.offset_joints, fit.offsets_rad, strict=True):
            if f"joint{j}" in old:
                lines.append(
                    f"joint{j} offset was {math.degrees(old[f'joint{j}']):+.2f} deg in the calibration file"
                )
    return "\n".join(lines)


def write_outputs(
    fit: HandEyeFit,
    out_dir: Path,
    *,
    source: str,
    calibration: Mapping[str, Any] | None,
    check: TouchCheck | None,
    intrinsics: Intrinsics,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    result: dict[str, Any] = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "side": fit.side,
        "source": "dimos.robot.manipulators.dual_openyam.calibration.handeye_solve",
        "captures": source,
        "intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "coeffs": list(intrinsics.coeffs),
            "model": intrinsics.model,
        },
        "fit": fit.to_json(),
    }
    if calibration is not None:
        base_in_rig = se3_from_pose(calibration["arms"][fit.side]["base_in_rig"])
        result["camera_in_rig"] = pose_to_dict(base_in_rig * fit.camera_in_base)
        result["base_in_rig_from"] = str(calibration.get("date", ""))
    if check is not None:
        result["touch_grid_check"] = check.to_json()
    json_path = out_dir / f"handeye_{fit.side}_{today}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# Dual OpenYAM hand-eye ({fit.side} arm)",
        "",
        f"Captures: `{source}`",
        "",
        "```",
        format_fit(fit, calibration),
        "```",
        "",
    ]
    if check is not None:
        lines += [
            f"## Held-out touch grid ({check.touches} touches, tool {check.tool_in_gripper_m})",
            "",
            f"{check.rms_mm:.1f} mm rms, worst {check.worst_mm:.1f} mm, mean offset "
            f"{np.round(check.mean_offset_mm, 1).tolist()} mm (camera frame)",
            "",
            "| point | dx | dy | dz | norm |",
            "|---|---|---|---|---|",
            *(
                f"| {r['point']} | {r['error_mm'][0]:+.1f} | {r['error_mm'][1]:+.1f} | {r['error_mm'][2]:+.1f} | {r['norm_mm']:.1f} |"
                for r in check.per_touch_mm
            ),
            "",
        ]
    lines.append(f"Result JSON: `{json_path}`")
    md_path = out_dir / f"handeye_report_{fit.side}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def table_board_from_frames(folder: Path, intrinsics: Intrinsics) -> pin.SE3:
    source = SavedFrames.from_dir(
        folder, "color_*.png" if any(folder.glob("color_*.png")) else "*.png"
    )
    sighting = sight_board(source, TABLE_BOARD, intrinsics, count=len(source.paths))
    if sighting.pose is None:
        raise RuntimeError(f"the table board is not visible in {folder}")
    return sighting.pose.board_in_camera


app = typer.Typer(add_completion=False)


@app.command()
def main(
    captures: Path = typer.Argument(..., help="Captures JSON or recordings/<run-id>"),
    side: str = typer.Option("right", help="Which arm: left or right"),
    calibration: Path = typer.Option(
        DEFAULT_CALIBRATION, help="Workcell calibration JSON (intrinsics, base poses)"
    ),
    offset_joints: str = typer.Option("2,3,4,5", help="Joints whose zero offset is fitted"),
    touch_grid: Path | None = typer.Option(
        None, help="touch_grid_<side>.json for the held-out check"
    ),
    touch_frames: Path | None = typer.Option(
        None, help="Overhead PNGs of the table board at those touches"
    ),
    tool: str = typer.Option("pin", help="Tool point of the touches: pin or fingertip"),
    out: Path | None = typer.Option(None, help="Output folder (default: next to the captures)"),
) -> None:
    """Fit camera pose, board-on-gripper and joint zero offsets from gripper-board captures."""
    if side not in SIDES:
        raise typer.BadParameter(f"side must be left or right, got {side!r}")
    cal = json.loads(calibration.read_text())
    intrinsics = Intrinsics.from_calibration(cal)
    kinematics = ArmKinematics()
    records = load_records(captures, side)
    typer.echo(f"{len(records)} capture(s) from {captures}")
    joints = tuple(int(j) for j in offset_joints.split(",") if j)
    fit = fit_handeye(
        side, records, kinematics=kinematics, intrinsics=intrinsics, offset_joints=joints
    )
    typer.echo(format_fit(fit, cal))
    check = None
    if touch_grid is not None and touch_frames is not None:
        tool_point = PIN_TIP_IN_GRIPPER_M if tool == "pin" else FINGERTIP_IN_GRIPPER_M
        check = touch_check(
            fit,
            json.loads(touch_grid.read_text()),
            table_board_from_frames(touch_frames, intrinsics),
            kinematics=kinematics,
            tool_in_gripper=tool_point,
        )
        typer.echo(
            f"held-out touch grid ({check.touches} touches, {tool}): {check.rms_mm:.1f} mm rms, "
            f"worst {check.worst_mm:.1f} mm, mean offset {np.round(check.mean_offset_mm, 1).tolist()} mm"
        )
    out_dir = out or (captures if captures.is_dir() else captures.parent)
    json_path, md_path = write_outputs(
        fit, out_dir, source=str(captures), calibration=cal, check=check, intrinsics=intrinsics
    )
    typer.echo(f"wrote {json_path}\n      {md_path}")


if __name__ == "__main__":
    app()
