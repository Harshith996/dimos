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

"""Deployment data shared by simulation, camera acquisition, IK and display."""

from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_SIDES
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.sensors.spec import Mount


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SpatialStation:
    scene: Path
    bases: dict[str, Mount]
    workspace: Camera
    wrists: dict[str, Camera]

    def __post_init__(self) -> None:
        if set(self.bases) != set(DUAL_OPENYAM_SIDES) or set(self.wrists) != set(
            DUAL_OPENYAM_SIDES
        ):
            raise ValueError("station requires left and right bases and wrist cameras")
        if any(mount.link != "world" for mount in self.bases.values()):
            raise ValueError("station base poses must be world-relative")
        # The existing bimanual collector records a common video mode.
        modes = {(c.width, c.height, c.rate_hz) for c in self.wrists.values()}
        if len(modes) != 1:
            raise ValueError("the bimanual collector requires matching wrist video modes")
