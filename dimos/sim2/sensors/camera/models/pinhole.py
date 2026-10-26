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

"""Rectified pinhole calibration. Pixel coordinates refer to pixel centres."""

from __future__ import annotations

import math

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

    def resized(self, width: int, height: int) -> PinholeIntrinsics:
        """Resize the same image extent, preserving pixel-centre coordinates.

        This is image resampling, not calibration for another hardware mode.
        Hardware cropping/binning may require different measured intrinsics.
        """
        sx, sy = width / self.width, height / self.height
        return PinholeIntrinsics(
            width,
            height,
            self.fx * sx,
            self.fy * sy,
            (self.cx + 0.5) * sx - 0.5,
            (self.cy + 0.5) * sy - 0.5,
        )
