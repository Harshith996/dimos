# Copyright 2025-2026 Dimensional Inc.
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


"""Synthetic target recordings establish loader behavior, not human-data availability."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.rl.env import TrackingEnv
from dimos.robot.galaxea.r1pro.rl.model import ARM_JOINT_NAMES, WRIST_SITE_NAMES, load_model
from dimos.robot.galaxea.r1pro.rl.recorded_targets import RecordedTargetSource


def document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {
            "reference_frame": "world",
            "position_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "wxyz",
            "clock": "relative_seconds",
            "source": "synthetic-test",
            "kind": "synthetic_fixture",
            "calibration_id": "test-only",
            "session_id": "synthetic-session-1",
            "operator_id": None,
        },
        "frames": [
            {
                "stamp_s": 0.0,
                "positions_m": [[0.1, 0.2, 0.3], [-0.1, -0.2, 0.3]],
                "position_valid": [True, True],
                "orientation_valid": [False, False],
                "quaternions_wxyz": [None, None],
                "confidence": [0.5, 1.0],
            }
        ],
    }


def write_source(tmp_path: Path, data: dict[str, Any]) -> RecordedTargetSource:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(data))
    return RecordedTargetSource(path)


def test_loader_is_causal_and_preserves_timestamps_and_provenance(tmp_path: Path) -> None:
    data = document()
    later = deepcopy(data["frames"][0])
    later["stamp_s"] = 0.05
    later["positions_m"][0][0] = 0.9
    data["frames"].append(later)
    source = write_source(tmp_path, data)
    assert source.sample(-0.001) is None
    early = source.sample(0.049)
    assert early is not None
    assert early.stamp_s == 0.0
    assert early.positions_m[0, 0] == 0.1
    current = source.sample(0.05)
    assert current is not None
    assert current.stamp_s == 0.05
    assert current.positions_m[0, 0] == 0.9
    assert source.sample(10.0) is current
    assert current.provenance == "synthetic_fixture:synthetic-test"
    assert current.calibration_id == "test-only"
    np.testing.assert_array_equal(current.orientation_valid, [False, False])
    np.testing.assert_array_equal(source.confidence[0], [0.5, 1.0])
    assert source.metadata["session_id"] == "synthetic-session-1"


def test_end_of_recording_expires_in_real_environment(tmp_path: Path) -> None:
    source = write_source(tmp_path, document())
    env = TrackingEnv(load_model(), ARM_JOINT_NAMES, WRIST_SITE_NAMES)
    env.reset()
    for _ in range(7):
        _, _, terminated, _, info = env.step(np.zeros(14), source.sample(env.time_s))
    assert terminated is True
    assert info["termination_reason"] == "target_timeout"
    assert env.time_s == pytest.approx(0.12)
    # Orientation storage placeholders never enter actor observations as targets.
    np.testing.assert_array_equal(env.observation()[48:56], np.zeros(8))


def test_optional_preferences_and_partial_orientation_are_preserved(tmp_path: Path) -> None:
    data = document()
    row = data["frames"][0]
    row["elbow_positions_m"] = [[0.1, 0.2, 0.5], [0.1, -0.2, 0.5]]
    row["chest_quaternion_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    row["quaternions_wxyz"] = [[0.0, 1.0, 0.0, 0.0], None]
    row["orientation_valid"] = [True, False]
    source = write_source(tmp_path, data)
    frame = source.sample(0.0)
    assert frame is not None
    np.testing.assert_array_equal(frame.elbow_positions_m, row["elbow_positions_m"])
    np.testing.assert_array_equal(frame.chest_quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(frame.orientation_valid, [True, False])


@pytest.mark.parametrize(
    "fault",
    [
        "body_frame",
        "degrees",
        "nonmonotonic",
        "missing_orientation",
        "bad_confidence",
        "string_mask",
    ],
)
def test_loader_rejects_ambiguous_or_inconsistent_recordings(tmp_path: Path, fault: str) -> None:
    data = document()
    if fault == "body_frame":
        data["metadata"]["reference_frame"] = "body"
    elif fault == "degrees":
        data["metadata"]["angle_unit"] = "deg"
    elif fault == "nonmonotonic":
        data["frames"].append(deepcopy(data["frames"][0]))
    elif fault == "missing_orientation":
        data["frames"][0]["orientation_valid"][0] = True
    elif fault == "bad_confidence":
        data["frames"][0]["confidence"] = [1.1, 1.0]
    else:
        data["frames"][0]["position_valid"] = ["false", True]
    with pytest.raises(ValueError):
        write_source(tmp_path, data)
