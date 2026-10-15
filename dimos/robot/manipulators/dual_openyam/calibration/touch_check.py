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

"""Procedure step 5: both arms agree on where a point on the table is.

Each arm touches the same marked cross with the end of its closed fingertips.
The two fingertip positions, computed from each arm's joint angles, must agree
within 5 mm. A sideways mismatch points at the base spacing; one that grows
with distance from the bases points at a turned base. The fingertip height is
also compared with the table height, which tests the finger length, the table
measurement and the joint zeros of the arm together.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from dimos.robot.manipulators.dual_openyam.calibration.encoders import SIDES
from dimos.robot.manipulators.dual_openyam.calibration.kinematics import RigGeometry

TOUCH_TOLERANCE_M = 0.005


@dataclass(frozen=True)
class TouchResult:
    fingertips_m: Mapping[str, np.ndarray]
    table_z_m: float

    @property
    def difference_m(self) -> np.ndarray:
        """Left fingertip minus right fingertip, rig frame."""
        return self.fingertips_m["left"] - self.fingertips_m["right"]

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.difference_m))

    @property
    def passed(self) -> bool:
        return self.distance_m <= TOUCH_TOLERANCE_M

    def height_above_table_m(self, side: str) -> float:
        return float(self.fingertips_m[side][2] - self.table_z_m)


def touch_record(
    side: str,
    joints_rad: np.ndarray,
    fingertip_m: np.ndarray,
    geometry: RigGeometry,
    taken_at: datetime,
) -> dict[str, Any]:
    return {
        "step": "both_arms_agree",
        "side": side,
        "taken_at": taken_at.isoformat(timespec="seconds"),
        "joints_rad": [float(v) for v in joints_rad],
        "fingertip_rig_m": [float(v) for v in fingertip_m],
        "geometry": {
            **asdict(geometry),
            "finger_length_m": dict(geometry.finger_length_m),
        },
    }


def save_touch(session_dir: Path, record: Mapping[str, Any]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"touch_{record['side']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_touches(session_dir: Path) -> dict[str, dict[str, Any]]:
    touches = {}
    for side in SIDES:
        path = session_dir / f"touch_{side}.json"
        if path.is_file():
            touches[side] = json.loads(path.read_text())
    return touches


def format_fingertip(side: str, fingertip_m: np.ndarray, table_z_m: float) -> str:
    x, y, z = fingertip_m * 1000.0
    return (
        f"{side:<5} fingertip  x {x:+7.1f}  y {y:+7.1f}  z {z:+7.1f} mm"
        f"   ({z - table_z_m * 1000.0:+.1f} mm above the table)"
    )


def format_touch(result: TouchResult) -> str:
    dx, dy, dz = result.difference_m * 1000.0
    lines = [format_fingertip(side, result.fingertips_m[side], result.table_z_m) for side in SIDES]
    lines += [
        "",
        f"left - right   x {dx:+.1f}  y {dy:+.1f}  z {dz:+.1f} mm   distance {result.distance_m * 1000:.1f} mm",
        f"Pass: {TOUCH_TOLERANCE_M * 1000:.0f} mm or less.",
        (
            "Step 5 PASSED: both arms agree on the cross."
            if result.passed
            else "Step 5 FAILED: the arms disagree. Sideways (y) points at the base spacing; "
            "forward (x) at a turned base; height (z) at a finger length or joint zero."
        ),
    ]
    return "\n".join(lines)
