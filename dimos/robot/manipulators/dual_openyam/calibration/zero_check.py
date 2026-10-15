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

"""Procedure step 1: the joint zeros hold across power cycles.

Both arms park on their rest supports and every joint is read; the arms are
power-cycled and read again, three times in all. A joint passes when its
readings agree within 0.2 deg. The mean reading is reported too: the model's
zero configuration is the rest pose, so it is the joint's offset at rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from dimos.robot.manipulators.dual_openyam.calibration.encoders import (
    JOINT_COUNT,
    SIDES,
    EncoderReading,
)

ZERO_HOLD_TOLERANCE_DEG = 0.2
ZERO_HOLD_CYCLES = 3
JOINT_NAMES = tuple(f"{side}_joint{index}" for side in SIDES for index in range(1, JOINT_COUNT + 1))
_SNAPSHOT_PATTERN = re.compile(r"zero_cycle(\d+)\.json")


@dataclass(frozen=True)
class ZeroHoldResult:
    """Readings in degrees, one row per power cycle, one column per joint."""

    cycles: tuple[int, ...]
    readings_deg: np.ndarray

    @property
    def spread_deg(self) -> np.ndarray:
        return np.ptp(self.readings_deg, axis=0)

    @property
    def mean_deg(self) -> np.ndarray:
        return self.readings_deg.mean(axis=0)

    @property
    def joint_passed(self) -> np.ndarray:
        return self.spread_deg <= ZERO_HOLD_TOLERANCE_DEG

    @property
    def complete(self) -> bool:
        return len(self.cycles) >= ZERO_HOLD_CYCLES

    @property
    def passed(self) -> bool:
        return self.complete and bool(self.joint_passed.all())


def snapshot_record(reading: EncoderReading, cycle: int, taken_at: datetime) -> dict[str, Any]:
    return {
        "step": "zero_hold",
        "cycle": cycle,
        "taken_at": taken_at.isoformat(timespec="seconds"),
        "samples": reading.samples,
        "max_spread_deg": float(np.degrees(reading.max_spread_rad)),
        "joints_rad": {
            f"{side}_joint{index + 1}": value
            for side in SIDES
            for index, value in enumerate(reading.arms[side].joints_rad)
        },
        "gripper_motor_rad": {side: reading.arms[side].gripper_motor_rad for side in SIDES},
        "temperatures_c": {side: list(reading.arms[side].temperatures_c) for side in SIDES},
    }


def save_snapshot(session_dir: Path, record: Mapping[str, Any]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"zero_cycle{record['cycle']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_snapshots(session_dir: Path) -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    if not session_dir.is_dir():
        return snapshots
    for path in session_dir.iterdir():
        match = _SNAPSHOT_PATTERN.fullmatch(path.name)
        if match:
            snapshots[int(match.group(1))] = json.loads(path.read_text())
    return dict(sorted(snapshots.items()))


def evaluate_zero_hold(snapshots: Mapping[int, Mapping[str, Any]]) -> ZeroHoldResult:
    cycles = tuple(sorted(snapshots))
    if not cycles:
        raise ValueError("no zero-hold snapshots")
    readings = np.degrees(
        np.array(
            [[snapshots[cycle]["joints_rad"][joint] for joint in JOINT_NAMES] for cycle in cycles],
            dtype=np.float64,
        )
    )
    return ZeroHoldResult(cycles=cycles, readings_deg=readings)


def format_zero_hold(result: ZeroHoldResult) -> str:
    """A table a person reads at the rig: one row per joint."""
    header = (
        f"{'joint':<14}"
        + "".join(f"{'cycle ' + str(cycle):>10}" for cycle in result.cycles)
        + f"{'spread':>9}{'result':>8}"
    )
    lines = [header, "-" * len(header)]
    for column, joint in enumerate(JOINT_NAMES):
        values = "".join(
            f"{result.readings_deg[row, column]:>+10.2f}" for row in range(len(result.cycles))
        )
        spread = result.spread_deg[column]
        if len(result.cycles) < 2:
            verdict = ""
        else:
            verdict = "ok" if result.joint_passed[column] else "FAIL"
        lines.append(f"{joint:<14}{values}{spread:>9.2f}{verdict:>8}")
    lines.append("")
    lines.append(f"Readings in degrees. Pass: every spread {ZERO_HOLD_TOLERANCE_DEG} deg or less.")
    if not result.complete:
        remaining = ZERO_HOLD_CYCLES - len(result.cycles)
        lines.append(f"{remaining} more power cycle(s) to go.")
    elif result.passed:
        lines.append("Step 1 PASSED: every zero holds across power cycles.")
    else:
        failed = [
            joint for joint, ok in zip(JOINT_NAMES, result.joint_passed, strict=True) if not ok
        ]
        lines.append("Step 1 FAILED on " + ", ".join(failed) + ". Stop and find out why.")
    return "\n".join(lines)
