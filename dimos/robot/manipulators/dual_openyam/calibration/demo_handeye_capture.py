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

"""Automatic hand-eye capture: one arm shows the gripper board on the jig to the overhead camera.

    # terminal 1 (rig)
    dimos run dual-openyam-step-test-camera --left-can-port can0 --right-can-port can1 \\
        --inertial-overrides dimos/robot/manipulators/dual_openyam/calibration/gravity_both_2026-10-16.json \\
        --listen-host 0.0.0.0 --rerun-open none --rerun-web
    # terminal 2
    python -m dimos.robot.manipulators.dual_openyam.calibration.demo_handeye_capture --side right
    python -m dimos.robot.manipulators.dual_openyam.calibration.handeye_solve <captures.json> --side right
    # mock: dimos run dual-openyam-step-test, then add
    #   --no-confirm --frames ~/.local/share/dimos/calibration/dual_openyam/2026-10-14/handeye/frames

The arm ramps through the poses of ``handeye_poses`` (board facing the camera
within 45 deg) at 10 deg/s, waits 1 s at each, grabs three frames, keeps the
best gripper-board detection (>= 6 corners) with the measured joint angles,
appends it to ``captures_<side>.json`` in the output folder and drops a
``gravity_events`` marker carrying the same record, so the recording holds
everything ``handeye_solve`` needs. Progress and the running count of good
poses print as it goes. Enter before each pose unless ``--no-confirm``; ``s``
skips a pose, ``p`` (or Ctrl-C) parks the arms and ends the run. The other arm
stays where the module holds it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from functools import partial
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import typer

from dimos.porcelain.dimos import Dimos
from dimos.robot.manipulators.dual_openyam.calibration.board_vision import (
    GRIPPER_BOARD,
    FrameSource,
    Intrinsics,
    SavedFrames,
    StreamFrames,
    pose_to_dict,
    sight_board,
)
from dimos.robot.manipulators.dual_openyam.calibration.encoders import SIDES
from dimos.robot.manipulators.dual_openyam.calibration.gravity_poses import (
    ArmModel,
    Pose,
    home_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.handeye_poses import (
    DEFAULT_CALIBRATION,
    DEFAULT_POSE_COUNT,
    MAX_FACING_DEG,
    JigBoard,
    format_pose_set,
    generate_handeye_poses,
)
from dimos.robot.manipulators.dual_openyam.calibration.handeye_solve import (
    CAPTURE_LABEL,
    HandEyeRecord,
    save_records,
)
from dimos.robot.manipulators.dual_openyam.calibration.progress import (
    ParkRequestedError,
    confirm,
    countdown,
)

RPC_OVERHEAD_S = 0.3
STILL_DEG = 0.1
DEFAULT_OUT_ROOT = Path.home() / ".local/share/dimos/calibration/dual_openyam"

app = typer.Typer(add_completion=False)


def travel_s(a: Sequence[float], b: Sequence[float], speed_deg_s: float) -> float:
    return math.degrees(max(abs(x - y) for x, y in zip(a, b, strict=True))) / speed_deg_s


def measured_rad(status: dict[str, Any], joint_names: Sequence[str]) -> np.ndarray:
    return np.asarray([math.radians(status["joints"][n]["measured_deg"]) for n in joint_names])


def path_to(arm: ArmModel, current: Sequence[float], target: Pose) -> list[Pose] | None:
    """Direct ramp when it clears the table, otherwise via home, otherwise None."""
    if arm.transition_ok(current, target.joints_rad):
        return [target]
    home = home_pose(arm.side)
    if arm.transition_ok(current, home.joints_rad) and arm.transition_ok(
        home.joints_rad, target.joints_rad
    ):
        return [home, target]
    return None


OVERHEAD_SERIAL = "230322272156"
UV = "/home/anvil/.local/bin/uv"


class SnapFrames:
    """Fixed-exposure frames grabbed by realsense_snap.py through uv, one camera open per call."""

    def __init__(self, serial: str, exposure_us: int, keep_dir: Path | None = None) -> None:
        self.serial, self.exposure_us, self.keep_dir = serial, exposure_us, keep_dir

    def grab(self, count: int = 3, timeout_s: float = 30.0) -> list[np.ndarray]:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    UV,
                    "run",
                    "--no-project",
                    "--with",
                    "pyrealsense2",
                    "--with",
                    "numpy",
                    "--with",
                    "pillow",
                    "--python",
                    "3.12",
                    "python",
                    str(Path(__file__).with_name("realsense_snap.py")),
                    self.serial,
                    tmp,
                    str(self.exposure_us),
                    str(count),
                ],
                check=True,
                capture_output=True,
                timeout=max(timeout_s, 30.0),
            )
            return [cv2.imread(str(p)) for p in sorted(Path(tmp).glob("color_*.png"))]

    def close(self) -> None:
        pass


def open_frames(frames_dir: Path | None, topic: str, exposure_us: int | None = None) -> FrameSource:
    if frames_dir is None and exposure_us is not None:
        typer.echo(
            f"camera: fixed exposure {exposure_us} us, grabbed per pose via realsense_snap.py"
        )
        return SnapFrames(OVERHEAD_SERIAL, exposure_us)
    if frames_dir is not None:
        pattern = "*.png"
        for candidate in ("right_*.png", "left_*.png", "color_*.png"):
            if any(frames_dir.glob(candidate)):
                pattern = candidate
                break
        source = SavedFrames.from_dir(frames_dir, pattern)
        typer.echo(f"camera: {len(source.paths)} saved frame(s) from {frames_dir}")
        return source
    stream = StreamFrames(topic)
    try:
        stream.grab(1, timeout_s=5.0)
    except TimeoutError:
        stream.close()
        raise typer.BadParameter(
            f"no frames on the {topic!r} stream; run dual-openyam-step-test-camera or pass --frames"
        ) from None
    typer.echo(f"camera: live frames on {topic!r}")
    return stream


@app.command()
def run(
    side: str = typer.Option("right", help="Which arm carries the board: left or right"),
    poses: int = typer.Option(DEFAULT_POSE_COUNT, help="Number of poses to visit"),
    frames: Path | None = typer.Option(
        None, "--frames", help="Saved PNGs to use instead of the camera stream (mock runs)"
    ),
    board_normal: str = typer.Option(
        "+z",
        help="Board normal in the gripper frame (+z = back along the fingers toward the wrist)",
    ),
    board_centre_m: str = typer.Option("0,0,-0.146", help="Board centre in the gripper frame, m"),
    max_facing_deg: float = typer.Option(
        MAX_FACING_DEG, help="Board may be this far from square-on"
    ),
    calibration: Path = typer.Option(DEFAULT_CALIBRATION, help="Provisional workcell calibration"),
    out: Path | None = typer.Option(None, help="Output folder (captures JSON, frames)"),
    speed_deg_s: float = typer.Option(10.0, help="Ramp speed"),
    settle_s: float = typer.Option(1.0, help="Wait at each pose before the camera looks"),
    camera_topic: str = typer.Option("color_image", help="Camera stream name"),
    exposure_us: int | None = typer.Option(
        1200,
        help="Grab frames directly at this fixed exposure (the dimos camera stream is auto-only "
        "and blows out under the rig lights); 0 = use the camera stream instead",
    ),
    confirm_each: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Ask before each pose (Enter / s skip / p park)"
    ),
) -> None:
    """Ramp one arm through board-facing poses and record the gripper board at each."""
    if side not in SIDES:
        raise typer.BadParameter(f"side must be left or right, got {side!r}")
    cal = json.loads(calibration.read_text())
    intrinsics = Intrinsics.from_calibration(cal)
    jig = JigBoard.from_options(board_normal, [float(v) for v in board_centre_m.split(",")])
    arm = ArmModel(side=side)
    typer.echo(
        f"generating {side} arm poses (board normal {board_normal}, centre {board_centre_m} m) ..."
    )
    pose_set = generate_handeye_poses(
        side, calibration=cal, jig=jig, count=poses, max_facing_deg=max_facing_deg, arm=arm
    )
    typer.echo(format_pose_set(pose_set))
    plan = list(pose_set.poses)
    home = home_pose(side)
    out_dir = out or DEFAULT_OUT_ROOT / date.today().isoformat() / datetime.now().strftime(
        f"handeye_{side}_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    captures_path = out_dir / f"captures_{side}.json"
    typer.echo(f"output {out_dir}")
    frame_source = open_frames(frames, camera_topic, exposure_us or None)

    dimos = Dimos.connect()
    module: Any = dimos.JointStepTestModule
    records: list[HandEyeRecord] = []
    park_s = 120.0 / speed_deg_s + 2.0
    started_capture = False
    try:
        status: dict[str, Any] = module.status()
        typer.echo(
            f"connected: adapter={status['adapter']} loop={status['measured_hz']} Hz "
            f"ramping={status['ramping']}"
        )
        current = measured_rad(status, arm.joint_names)
        expected = travel_s(current, home.joints_rad, speed_deg_s) + sum(
            travel_s(a.joints_rad, b.joints_rad, speed_deg_s) + settle_s + 1.0 + RPC_OVERHEAD_S
            for a, b in zip([home, *plan], plan, strict=False)
        )
        typer.echo(
            f"plan: {side} arm, {len(plan)} pose(s), about {expected / 60:.1f} min of motion"
            + (" plus confirmations" if confirm_each else "")
        )
        if not confirm("Ramp both arms to the test pose (home)?", confirm_each):
            raise ParkRequestedError("home ramp skipped")
        typer.echo(
            countdown(
                "go home",
                travel_s(current, home.joints_rad, speed_deg_s),
                lambda: module.go_to_test_pose(speed_deg_s),
            )
        )
        current = np.asarray(home.joints_rad)
        for index, pose in enumerate(plan, 1):
            angles = " ".join(f"{v:.1f}" for v in pose.joints_deg)
            label = f"pose {index}/{len(plan)} {pose.name} [{angles}] deg"
            if not confirm(label, confirm_each):
                continue
            path = path_to(arm, current, pose)
            if path is None:
                typer.echo(f"{label}: no table-clearing ramp from the current pose; skipped")
                continue
            started_capture = True
            module.mark(
                f"handeye_pose_{index:02d}",
                {"index": index, "side": side, "targets_deg": list(pose.joints_deg)},
            )
            for waypoint in path:
                typer.echo(
                    countdown(
                        f"{label}: ramp to {waypoint.name}",
                        travel_s(current, waypoint.joints_rad, speed_deg_s),
                        partial(module.go_to, waypoint.as_targets(), speed_deg_s),
                    )
                )
                current = np.asarray(waypoint.joints_rad)
            countdown(f"{label}: settle", settle_s, partial(time.sleep, settle_s))
            before = measured_rad(module.status(), arm.joint_names)
            try:
                sighting = sight_board(frame_source, GRIPPER_BOARD, intrinsics)
            except TimeoutError as error:
                typer.echo(f"{label}: camera: {error}", err=True)
                continue
            after = measured_rad(module.status(), arm.joint_names)
            if np.degrees(np.abs(after - before)).max() > STILL_DEG:
                typer.echo(f"{label}: the arm moved during the capture; pose dropped")
                continue
            if sighting.detection is None or sighting.pose is None:
                typer.echo(f"{label}: board not seen (< 6 corners); {len(records)} good so far")
                continue
            frame = sighting.frames[sighting.frame_index]
            cv2.imwrite(str(out_dir / f"{side}_{index:02d}.png"), frame)
            record = HandEyeRecord(
                joints_rad=[float(v) for v in (before + after) / 2.0],
                detection=sighting.detection,
                name=pose.name,
                taken_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                rms_px=sighting.pose.rms_px,
            )
            records.append(record)
            save_records(captures_path, records)
            module.mark(
                CAPTURE_LABEL,
                {
                    "side": side,
                    "index": index,
                    "record": record.to_json(),
                    "board_in_camera": pose_to_dict(sighting.pose.board_in_camera),
                    "facing_deg": sighting.pose.facing_deg,
                },
            )
            typer.echo(
                f"{label}: {sighting.detection.count}/{GRIPPER_BOARD.corner_count} corners, "
                f"{sighting.pose.rms_px:.2f} px, board {sighting.pose.distance_m * 100:.0f} cm away, "
                f"{sighting.pose.facing_deg:.0f} deg off facing -> {len(records)} good of {index}"
            )
        typer.echo(
            f"capture done: {len(records)} good pose(s) in {captures_path}\n"
            "solve with: python -m dimos.robot.manipulators.dual_openyam.calibration.handeye_solve "
            f"{captures_path} --side {side}"
        )
    except ParkRequestedError as request:
        typer.echo(f"\n{request}: ending the run")
    except KeyboardInterrupt:
        typer.echo("\ninterrupted: ending the run")
    finally:
        if started_capture:
            try:
                module.mark("handeye_end", {"side": side, "good_poses": len(records)})
            except Exception as error:
                typer.echo(f"end marker failed: {error}", err=True)
        try:
            typer.echo(countdown("park on the supports", park_s, lambda: module.park(speed_deg_s)))
        except Exception as error:
            typer.echo(f"park failed: {error}; the arm holds its last target", err=True)
        frame_source.close()
        dimos.stop()


if __name__ == "__main__":
    app()
