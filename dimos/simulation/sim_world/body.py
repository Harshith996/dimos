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

"""What the simulated world needs from a robot: how to build it, and how to drive it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.msgs.sim_msgs.Contacts import Part
from dimos.simulation.sensors.mid360.lidar import Raycaster, SimMid360


def _untuned(model: mujoco.MjModel) -> None:
    pass


@dataclass(frozen=True)
class Rig:
    """A robot as a scene is built around it: its model, its Mid-360 mount and its driver."""

    spec: Callable[[], mujoco.MjSpec]
    base_body: str
    mount_xyz: NDArray[np.float64]
    mount_r: NDArray[np.float64]
    lidar: Callable[[Raycaster, int], SimMid360]
    command_timeout: float
    tune: Callable[[mujoco.MjModel], None] = _untuned


class Body(Protocol):
    """A robot in a compiled model, stepped one control tick at a time from a velocity command."""

    @property
    def dt(self) -> float: ...

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def reset(self, x: float, y: float, z_feet: float, yaw: float) -> None: ...

    def tick(
        self, command: NDArray[np.float64], on_substep: Callable[[], None] | None = None
    ) -> None: ...

    def base_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def base_velocity(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linear velocity in the world frame and angular velocity in the base frame."""
        ...

    def joint_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def part(self, geom: int) -> Part: ...
