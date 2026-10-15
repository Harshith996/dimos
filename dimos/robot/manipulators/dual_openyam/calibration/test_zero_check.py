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

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from dimos.robot.manipulators.dual_openyam.calibration.encoders import (
    ArmReading,
    EncoderReadError,
    EncoderReading,
    require_quiet_buses,
)
from dimos.robot.manipulators.dual_openyam.calibration.zero_check import (
    JOINT_NAMES,
    evaluate_zero_hold,
    format_zero_hold,
    load_snapshots,
    save_snapshot,
    snapshot_record,
)


def _reading(offsets_deg: list[float]) -> EncoderReading:
    radians = np.radians(offsets_deg)
    return EncoderReading(
        arms={
            "left": ArmReading(tuple(radians[:6]), -0.01, (30,) * 7),
            "right": ArmReading(tuple(radians[6:]), 0.02, (31,) * 7),
        },
        samples=20,
        max_spread_rad=0.0,
    )


def _capture(session: Path, cycle: int, offsets_deg: list[float]) -> None:
    taken_at = datetime(2026, 10, 14, 12, cycle, tzinfo=timezone.utc)
    save_snapshot(session, snapshot_record(_reading(offsets_deg), cycle, taken_at))


def test_three_matching_cycles_pass(tmp_path: Path) -> None:
    for cycle in (1, 2, 3):
        _capture(tmp_path, cycle, [0.05 * cycle] * 12)

    result = evaluate_zero_hold(load_snapshots(tmp_path))

    assert result.cycles == (1, 2, 3)
    assert result.passed
    np.testing.assert_allclose(result.spread_deg, 0.1)
    np.testing.assert_allclose(result.mean_deg, 0.1)
    assert "PASSED" in format_zero_hold(result)


def test_one_joint_that_slips_fails_and_is_named(tmp_path: Path) -> None:
    _capture(tmp_path, 1, [0.0] * 12)
    _capture(tmp_path, 2, [0.0] * 12)
    slipped = [0.0] * 12
    slipped[JOINT_NAMES.index("right_joint2")] = 0.5
    _capture(tmp_path, 3, slipped)

    result = evaluate_zero_hold(load_snapshots(tmp_path))

    assert not result.passed
    assert [j for j, ok in zip(JOINT_NAMES, result.joint_passed, strict=True) if not ok] == [
        "right_joint2"
    ]
    assert "FAILED on right_joint2" in format_zero_hold(result)


def test_partial_session_is_not_a_pass(tmp_path: Path) -> None:
    _capture(tmp_path, 1, [0.0] * 12)

    result = evaluate_zero_hold(load_snapshots(tmp_path))

    assert not result.complete
    assert not result.passed
    assert "2 more power cycle(s)" in format_zero_hold(result)


def test_snapshot_keeps_raw_negative_readings(tmp_path: Path) -> None:
    # The rest pose sits on the lower limits of joints 2 and 3; a reading just
    # below zero is the offset this step exists to find and must survive.
    offsets = [0.0] * 12
    offsets[JOINT_NAMES.index("left_joint2")] = -0.3
    _capture(tmp_path, 1, offsets)

    result = evaluate_zero_hold(load_snapshots(tmp_path))

    assert result.readings_deg[0, JOINT_NAMES.index("left_joint2")] == pytest.approx(-0.3)


def test_quiet_buses_pass() -> None:
    require_quiet_buses(
        {"left": "follower_l", "right": "follower_r"},
        frame_count=lambda port: 100,
        sleep=lambda _: None,
    )


def test_busy_bus_is_refused() -> None:
    counts = {"follower_l": iter([100, 100]), "follower_r": iter([5, 905])}

    with pytest.raises(EncoderReadError, match="follower_r"):
        require_quiet_buses(
            {"left": "follower_l", "right": "follower_r"},
            frame_count=lambda port: next(counts[port]),
            sleep=lambda _: None,
        )
