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

"""TEMP: one hand-eye capture of a teleoperated arm (joints from the teleop stack + one camera grab).

    python -m ...handeye_snap --side right --out DIR

Appends to DIR/captures_<side>.json in the format handeye_solve reads.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
import typer

from dimos.robot.manipulators.dual_openyam.calibration.board_vision import (
    GRIPPER_BOARD,
    Intrinsics,
    best_detection,
    solve_board_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.handeye_solve import (
    HandEyeRecord,
    load_records,
    save_records,
)
from dimos.robot.manipulators.dual_openyam.calibration.joint_stream import JointStream

CALIBRATION = Path(__file__).with_name("dual_openyam_calibration_2026-10-15.json")


def main(
    side: str = typer.Option(..., help="left or right"),
    out: Path = typer.Option(..., help="Session folder"),
    camera_url: str = typer.Option(
        "http://127.0.0.1:8090", help="camera_server.py (fixed exposure)"
    ),
    topic: str = typer.Option("coordinator_joint_state"),
) -> None:
    names = [f"{side}_joint{i}" for i in range(1, 7)]
    stream = JointStream(topic)
    try:
        before = stream.capture(names)
        frames = []
        for _ in range(3):
            data = urllib.request.urlopen(f"{camera_url}/raw.png", timeout=5).read()
            frames.append(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
            time.sleep(0.1)
        after = stream.capture(names)
    finally:
        stream.close()
    if np.degrees(np.abs(after - before)).max() > 0.1:
        raise SystemExit("the arm moved during the camera grab; hold still and capture again")
    intrinsics = Intrinsics.from_calibration(json.loads(CALIBRATION.read_text()))
    detection, index = best_detection(frames, GRIPPER_BOARD, intrinsics)
    if detection is None:
        raise SystemExit("gripper board not seen (fewer than 6 corners); turn it toward the camera")
    pose = solve_board_pose(detection, intrinsics)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"captures_{side}.json"
    records = load_records(path, side) if path.exists() else []
    name = f"{side}_{len(records) + 1:02d}"
    cv2.imwrite(str(out / f"{name}.png"), frames[index])
    records.append(
        HandEyeRecord(
            joints_rad=[float(v) for v in (before + after) / 2.0],
            detection=detection,
            name=name,
            taken_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            rms_px=pose.rms_px,
        )
    )
    save_records(path, records)
    typer.echo(
        f"{name}: {len(detection.object_points)} corners, {pose.rms_px:.2f} px, board {pose.distance_m * 1000:.0f} mm "
        f"from the camera, facing {pose.facing_deg:.0f} deg; {len(records)} {side} capture(s)"
    )


if __name__ == "__main__":
    typer.run(main)
