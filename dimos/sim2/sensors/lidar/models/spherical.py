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

"""Ideal instantaneous spherical scanner; no rolling-scan fidelity claim."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Spherical:
    columns: int = 360
    rows: int = 32
    elevation_min: float = -30.0
    elevation_max: float = 55.0
    min_range: float = 0.15
    max_range: float = 30.0

    def directions(self) -> NDArray[np.float64]:
        azimuth, elevation = np.meshgrid(
            np.linspace(-np.pi, np.pi, self.columns, endpoint=False),
            np.deg2rad(np.linspace(self.elevation_min, self.elevation_max, self.rows)),
        )
        return np.ascontiguousarray(
            np.column_stack(
                (
                    (np.cos(elevation) * np.cos(azimuth)).ravel(),
                    (np.cos(elevation) * np.sin(azimuth)).ravel(),
                    np.sin(elevation).ravel(),
                )
            )
        )
