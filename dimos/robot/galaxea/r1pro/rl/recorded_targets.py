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


"""Read already-retargeted user-owned wrist targets without future interpolation."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.rl.targets import TargetFrame


def _mask(value: Any, name: str) -> NDArray[np.bool_]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not bool for item in value)
    ):
        raise ValueError(f"{name} must contain exactly two JSON booleans")
    return np.asarray(value, dtype=bool)


class RecordedTargetSource:
    """Causal zero-order hold of relative-time targets, with original timestamps.

    Inputs already use calibrated robot world coordinates. This loader does not
    retarget human skeletons or synthesize robot actions. End-of-file retains the
    last original timestamp so the environment's independent watchdog can expire.
    """

    def __init__(self, path: str | Path) -> None:
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        if (
            not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
        ):
            raise ValueError("Target stream requires schema_version 1")
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Target stream requires metadata")
        for key, expected in {
            "reference_frame": "world",
            "position_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "wxyz",
            "clock": "relative_seconds",
        }.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Target stream metadata {key} must equal {expected!r}")
        for key in ("source", "calibration_id", "session_id"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"Target stream requires nonempty metadata {key}")
        if metadata.get("kind") not in ("user_owned", "synthetic_fixture"):
            raise ValueError("Target stream kind must be user_owned or synthetic_fixture")
        rows = document.get("frames")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Target stream requires at least one frame")
        frames: list[TargetFrame] = []
        confidence: list[NDArray[np.float64]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each target frame must be a JSON object")
            stamp = row.get("stamp_s")
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
                raise ValueError("Frame timestamps must be numbers")
            if not np.isfinite(stamp) or stamp < 0:
                raise ValueError("Frame timestamps must be finite nonnegative relative seconds")
            if frames and stamp <= frames[-1].stamp_s:
                raise ValueError("Frame timestamps must be strictly increasing")
            valid = row.get("valid", True)
            if type(valid) is not bool:
                raise ValueError("Frame valid must be a JSON boolean")
            position_valid = _mask(row.get("position_valid"), "position_valid")
            orientation_valid = _mask(row.get("orientation_valid"), "orientation_valid")
            quats = row.get("quaternions_wxyz", [None, None])
            if not isinstance(quats, list) or len(quats) != 2:
                raise ValueError("quaternions_wxyz must contain two wrist values or nulls")
            values = []
            for index, quat in enumerate(quats):
                if quat is None:
                    if orientation_valid[index]:
                        raise ValueError("Missing orientation must have orientation_valid false")
                    # Storage placeholder is masked before actor, reward and evaluation.
                    values.append([1.0, 0.0, 0.0, 0.0])
                else:
                    values.append(quat)
            score = np.asarray(row.get("confidence"), dtype=np.float64)
            if (
                score.shape != (2,)
                or not np.isfinite(score).all()
                or np.any((score < 0) | (score > 1))
            ):
                raise ValueError("confidence must contain two finite values between zero and one")
            score.flags.writeable = False
            frame = TargetFrame(
                positions_m=np.asarray(row.get("positions_m"), dtype=np.float64),
                quaternions_wxyz=np.asarray(values, dtype=np.float64),
                stamp_s=float(stamp),
                valid=valid,
                position_valid=position_valid,
                orientation_valid=orientation_valid,
                elbow_positions_m=None
                if row.get("elbow_positions_m") is None
                else np.asarray(row["elbow_positions_m"], dtype=np.float64),
                chest_quaternion_wxyz=None
                if row.get("chest_quaternion_wxyz") is None
                else np.asarray(row["chest_quaternion_wxyz"], dtype=np.float64),
                provenance=f"{metadata['kind']}:{metadata['source']}",
                calibration_id=metadata["calibration_id"],
            )
            frames.append(frame)
            confidence.append(score)
        self.frames = tuple(frames)
        self.confidence = tuple(confidence)
        self.metadata: dict[str, Any] = metadata
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self._times = tuple(frame.stamp_s for frame in frames)

    def sample(self, now_s: float) -> TargetFrame | None:
        """Return the latest past/current frame, never interpolate or renew its stamp."""
        if not np.isfinite(now_s):
            raise ValueError("Sample time must be finite")
        index = bisect_right(self._times, now_s) - 1
        return None if index < 0 else self.frames[index]
