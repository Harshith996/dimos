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


"""Current-only targets and sampled kinematic curriculum, independent of reward."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _array(value: Array, shape: tuple[int, ...], name: str) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TargetFrame:
    """SI units, world-frame poses, left wrist then right wrist, quaternion wxyz."""

    positions_m: Array
    quaternions_wxyz: Array
    stamp_s: float
    valid: bool = True
    elbow_positions_m: Array | None = None
    chest_quaternion_wxyz: Array | None = None
    reference_frame: str = "world"
    position_valid: NDArray[np.bool_] = field(default_factory=lambda: np.ones(2, dtype=bool))
    orientation_valid: NDArray[np.bool_] = field(default_factory=lambda: np.ones(2, dtype=bool))
    provenance: str = "procedural_fk"
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.stamp_s) or self.reference_frame != "world":
            raise ValueError("Target timestamp must be finite and reference frame must be world")
        object.__setattr__(self, "positions_m", _array(self.positions_m, (2, 3), "positions_m"))
        for name in ("position_valid", "orientation_valid"):
            mask = np.array(getattr(self, name), dtype=bool, copy=True)
            if mask.shape != (2,):
                raise ValueError(f"{name} must have shape (2,)")
            mask.flags.writeable = False
            object.__setattr__(self, name, mask)
        quats = _array(self.quaternions_wxyz, (2, 4), "quaternions_wxyz")
        if not np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6):
            raise ValueError("Wrist quaternions must have unit norm")
        object.__setattr__(self, "quaternions_wxyz", quats)
        if self.elbow_positions_m is not None:
            object.__setattr__(
                self, "elbow_positions_m", _array(self.elbow_positions_m, (2, 3), "elbows")
            )
        if self.chest_quaternion_wxyz is not None:
            chest = _array(self.chest_quaternion_wxyz, (4,), "chest")
            if not np.isclose(np.linalg.norm(chest), 1.0, atol=1e-6):
                raise ValueError("Chest quaternion must have unit norm")
            object.__setattr__(self, "chest_quaternion_wxyz", chest)


class TargetSource(Protocol):
    """External human-motion adapters retarget and stamp only the current frame."""

    def sample(self, now_s: float) -> TargetFrame | None: ...


class CurrentTargetBuffer:
    """Clock domain is supplied by the caller; stale/dropout frames never become fresh."""

    def __init__(self) -> None:
        self._latest: TargetFrame | None = None

    def push(self, frame: TargetFrame) -> None:
        if self._latest is not None and frame.stamp_s < self._latest.stamp_s:
            raise ValueError("Out-of-order target timestamp")
        self._latest = frame

    def reset(self) -> None:
        self._latest = None

    def current(self, now_s: float, max_age_s: float) -> TargetFrame | None:
        if not np.isfinite(now_s) or not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Clock must be finite and maximum age must be positive")
        if self._latest is None or not self._latest.valid:
            return None
        age = now_s - self._latest.stamp_s
        return self._latest if 0 <= age <= max_age_s else None


@dataclass(frozen=True)
class JointTrajectory:
    times_s: Array
    knots_rad: Array
    split: Literal["train", "heldout"]
    scenario: str
    seed: int

    def sample(self, now_s: float) -> Array:
        if not np.isfinite(now_s):
            raise ValueError("Sample time must be finite")
        t = float(np.clip(now_s, self.times_s[0], self.times_s[-1]))
        index = min(int(np.searchsorted(self.times_s, t, side="right")) - 1, len(self.times_s) - 2)
        u = (t - self.times_s[index]) / (self.times_s[index + 1] - self.times_s[index])
        blend = u**3 * (10 + u * (-15 + 6 * u))
        return np.asarray(
            self.knots_rad[index] + blend * (self.knots_rad[index + 1] - self.knots_rad[index]),
            dtype=np.float64,
        )


def generate_joint_trajectory(
    lower: Array,
    upper: Array,
    home: Array,
    *,
    seed: int,
    split: Literal["train", "heldout"],
    scenario: str,
    is_valid: Callable[[Array], bool],
    duration_s: float = 8.0,
    max_speed_rad_s: float = 0.2,
    margin_rad: float = 0.02,
    amplitude_rad: float = 0.08,
    validation_samples: int = 401,
) -> JointTrajectory:
    """Reject invalid paths; sampled collision checks do not certify swept safety.

    Train uses an excursion and return. Held-out scenarios add a pause or reversal.
    Distinct seed domains prevent accidental reuse when numeric seeds match.
    The validity callback must check this model's forbidden contacts and constraints.
    """
    home = np.asarray(home, dtype=np.float64)
    lower = _array(lower, home.shape, "lower")
    upper = _array(upper, home.shape, "upper")
    if (
        home.ndim != 1
        or not np.all(np.isfinite(home))
        or not np.isfinite([duration_s, max_speed_rad_s, margin_rad, amplitude_rad]).all()
        or duration_s <= 0
        or max_speed_rad_s <= 0
        or margin_rad < 0
        or amplitude_rad <= 0
        or validation_samples < 2
        or np.any(lower + margin_rad >= upper - margin_rad)
        or np.any(home < lower + margin_rad)
        or np.any(home > upper - margin_rad)
    ):
        raise ValueError("Invalid trajectory bounds, home, duration or sampling configuration")
    allowed = {"train": {"excursion"}, "heldout": {"pause", "reversal"}}
    if split not in allowed or scenario not in allowed[split]:
        raise ValueError("Scenario must belong to its declared split")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if split == "train" else 1]))
    count = 3 if split == "train" else 5
    times = np.linspace(0, duration_s, count)
    max_step = max_speed_rad_s * (duration_s / (count - 1)) / 1.875
    amplitude = min(amplitude_rad, max_step / 2)
    for _ in range(100):
        offset = rng.uniform(-amplitude, amplitude, home.shape)
        peak = np.clip(home + offset, lower + margin_rad, upper - margin_rad)
        reverse = np.clip(home - offset, lower + margin_rad, upper - margin_rad)
        knots = np.stack(
            [home, peak, home]
            if split == "train"
            else (
                [home, peak, peak, reverse, home]
                if scenario == "pause"
                else [home, peak, home, reverse, home]
            )
        )
        trajectory = JointTrajectory(times, knots, split, scenario, seed)
        if all(
            is_valid(trajectory.sample(float(t)))
            for t in np.linspace(0, duration_s, validation_samples)
        ):
            times.flags.writeable = False
            knots.flags.writeable = False
            return trajectory
    raise ValueError("No sampled collision-free trajectory found after 100 attempts")


class FkTrajectorySource:
    """Generate feasible pose references through FK, with optional fault intervals."""

    def __init__(
        self,
        trajectory: JointTrajectory,
        fk: Callable[[Array], TargetFrame],
        *,
        dropout_s: tuple[float, float] | None = None,
        stale_s: tuple[float, float] | None = None,
        stale_age_s: float = 1.0,
    ) -> None:
        for interval in (dropout_s, stale_s):
            if interval is not None and (
                not np.isfinite(interval).all() or not 0 <= interval[0] < interval[1]
            ):
                raise ValueError("Fault intervals must have finite increasing nonnegative times")
        if not np.isfinite(stale_age_s) or stale_age_s <= 0:
            raise ValueError("Stale age must be positive")
        self.trajectory = trajectory
        self.fk = fk
        self.dropout_s = dropout_s
        self.stale_s = stale_s
        self.stale_age_s = stale_age_s

    def sample(self, now_s: float) -> TargetFrame | None:
        if not np.isfinite(now_s):
            raise ValueError("Sample time must be finite")
        if self.dropout_s is not None and self.dropout_s[0] <= now_s < self.dropout_s[1]:
            return None
        stamp = now_s
        if self.stale_s is not None and self.stale_s[0] <= now_s < self.stale_s[1]:
            stamp -= self.stale_age_s
        return replace(self.fk(self.trajectory.sample(now_s)), stamp_s=stamp)


@dataclass(frozen=True)
class EvaluationCase:
    """Immutable evaluation cases are chosen before examining a trained policy."""

    seed: int
    scenario: str
    duration_s: float = 8.0
    dropout_s: tuple[float, float] | None = None
    stale_s: tuple[float, float] | None = None


HELDOUT_MANIFEST = (
    EvaluationCase(41001, "pause"),
    EvaluationCase(41002, "reversal"),
    EvaluationCase(41003, "pause", dropout_s=(3.0, 3.4)),
    EvaluationCase(41004, "reversal", stale_s=(3.0, 3.4)),
)


class FrozenTargetSource:
    """Replay recorded references verbatim; select only the latest available frame.

    The source holds the prior sampled reference between grid times. It never
    interpolates with a future frame. A recorded dropout remains a dropout.
    """

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        if len(samples) < 2:
            raise ValueError("Frozen targets require at least two samples")
        self.times_s = np.array([sample["query_time_s"] for sample in samples], dtype=float)
        if not np.isfinite(self.times_s).all() or np.any(np.diff(self.times_s) <= 0):
            raise ValueError("Frozen sample times must be finite and increasing")
        self.frames: tuple[TargetFrame | None, ...] = tuple(
            None if sample["frame"] is None else TargetFrame(**sample["frame"])
            for sample in samples
        )
        for query_time, frame in zip(self.times_s, self.frames, strict=True):
            if frame is not None and frame.stamp_s > query_time + 1e-9:
                raise ValueError("Frozen reference cannot contain a future target")
        self.times_s.flags.writeable = False

    def sample(self, now_s: float) -> TargetFrame | None:
        if not np.isfinite(now_s):
            raise ValueError("Sample time must be finite")
        if now_s < self.times_s[0] - 1e-9 or now_s > self.times_s[-1] + 1e-9:
            return None
        index = int(np.searchsorted(self.times_s, now_s + 1e-9, side="right")) - 1
        return self.frames[max(0, index)]
