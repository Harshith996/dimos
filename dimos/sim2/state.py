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

"""The running world's state in shared memory, so any local process can read it or drive actuators."""

from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


class SharedState:
    """A memory-mapped file laid out [sequence, time, qpos, qvel, sensordata, ctrl]; an odd sequence means mid-write."""

    def __init__(self, model: mujoco.MjModel, path: Path, create: bool = False) -> None:
        self.path = path
        sizes = [1, 1, model.nq, model.nv, model.nsensordata, model.nu]
        buffer = np.memmap(
            path, dtype=np.float64, mode="w+" if create else "r+", shape=(sum(sizes),)
        )
        offsets = np.cumsum([0, *sizes])
        self._sequence, self._time, self._qpos, self._qvel, self._sensordata, self.ctrl = (
            buffer[start:end] for start, end in pairwise(offsets)
        )
        self._mirror = buffer[offsets[1] : offsets[5]]

    def publish(self, data: mujoco.MjData) -> None:
        self._sequence[0] += 1
        self._time[0] = data.time
        self._qpos[:] = data.qpos
        self._qvel[:] = data.qvel
        self._sensordata[:] = data.sensordata
        self._sequence[0] += 1

    def read(self, data: mujoco.MjData) -> None:
        """Copy the latest consistent state into data (positions, velocities, sensor readings)."""
        while True:
            before = self._sequence[0]
            snapshot = self._mirror.copy()
            if before % 2 == 0 and self._sequence[0] == before:
                break
        nq, nv = len(self._qpos), len(self._qvel)
        data.time = snapshot[0]
        data.qpos[:] = snapshot[1 : 1 + nq]
        data.qvel[:] = snapshot[1 + nq : 1 + nq + nv]
        data.sensordata[:] = snapshot[1 + nq + nv :]


class SimClient:
    """A local copy of the running world: read its state, write its actuators by name."""

    def __init__(self, description: Mapping[str, Any]) -> None:
        self.model = mujoco.MjModel.from_binary_path(description["model"])
        self.data = mujoco.MjData(self.model)
        self.state = SharedState(self.model, Path(description["state"]))

    def read(self) -> mujoco.MjData:
        """Refresh the local data and its kinematics (body/site/camera poses)."""
        self.state.read(self.data)
        mujoco.mj_kinematics(self.model, self.data)
        return self.data

    def actuator_ids(self, names: list[str]) -> list[int]:
        return [self.model.actuator(name).id for name in names]

    def write(self, actuator_ids: list[int], ctrl: Any) -> None:
        self.state.ctrl[actuator_ids] = ctrl
