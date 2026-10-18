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

"""Body-joint snapshots received from a WebXR client."""

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

BodyTrackingMode: TypeAlias = Literal["off", "optional", "required"]
_FiniteFloat: TypeAlias = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_NonEmptyString: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r".*\S.*"),
]


class BodyJointPose(BaseModel):
    """One body joint's pose in the snapshot's WebXR reference space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: tuple[_FiniteFloat, _FiniteFloat, _FiniteFloat]
    orientation: tuple[_FiniteFloat, _FiniteFloat, _FiniteFloat, _FiniteFloat]


class BodyTrackingSnapshot(BaseModel):
    """Named body-joint poses captured in one WebXR reference space.

    ``joints=None`` means the body source is unavailable. An empty mapping
    means the source is available but did not resolve any joints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["body_tracking_snapshot"]
    capture_time_s: _FiniteFloat
    frame_id: _NonEmptyString
    joints: dict[_NonEmptyString, BodyJointPose] | None


class ArmPosture(BaseModel):
    """One operator arm's joint positions in the robot frame, from body tracking.

    The wrist and elbow fix the forearm direction that teleop IK uses as an
    elbow hint; the shoulder is carried for consumers that want the upper arm.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: _FiniteFloat
    hand: Literal["left", "right"]
    wrist: tuple[_FiniteFloat, _FiniteFloat, _FiniteFloat]
    elbow: tuple[_FiniteFloat, _FiniteFloat, _FiniteFloat]
    shoulder: tuple[_FiniteFloat, _FiniteFloat, _FiniteFloat] | None = None


# WebXR body-tracking joints describing each arm, in preference order. The
# wrist prefers the hand skeleton's wrist and falls back to the forearm twist
# joint beside it; "<side>-arm-upper" sits at the shoulder joint.
ARM_BODY_JOINTS: dict[str, dict[str, tuple[str, ...]]] = {
    side: {
        "shoulder": (f"{side}-arm-upper",),
        "elbow": (f"{side}-arm-lower",),
        "wrist": (f"{side}-hand-wrist", f"{side}-hand-wrist-twist"),
    }
    for side in ("left", "right")
}
