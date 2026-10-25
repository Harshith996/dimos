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

"""Position-only sim2 device for the existing manipulator coordinator."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock, local
import time
from typing import Literal, Protocol

import numpy as np

from dimos.hardware.manipulators.spec import ControlMode, ManipulatorInfo
from dimos.hardware.spec import JointLimits
from dimos.robot.manipulators.openyam.simulation import PUBLIC_JOINTS, YamArtifact
from dimos.sim2.client import SimRobotClient
from dimos.sim2.messages import StateFrame, StreamDescriptor

AuthorityMode = Literal["hold", "human", "correction", "policy"]


@dataclass(frozen=True)
class AcceptedTarget:
    sequence: int
    timestamp: float
    positions: tuple[float, ...]
    authority: AuthorityMode
    segment: int
    generation: int


class LearningControlDevice(Protocol):
    """Device operations shared by the sim2 and calibrated CAN boundaries."""

    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def activate(self) -> bool: ...
    def get_limits(self) -> JointLimits: ...
    def transition(self, mode: AuthorityMode) -> int: ...
    def observed_generation(self) -> int: ...


class _TickObservation(local):
    frame: StateFrame | None = None
    epoch: int = -1


class YamPositionDevice:
    """One writable client, owned and ticked by ControlCoordinator.

    A transition invalidates a tick that already read its input. Accepted targets
    are retained until feedback acknowledges their exact sim2 sequence.
    """

    def __init__(self, descriptor: StreamDescriptor, artifact: YamArtifact) -> None:
        self._client = SimRobotClient(descriptor, writable=True)
        self._joints = artifact.robot_instance().config.joints
        self._lock = RLock()
        self._connected = False
        self._enabled = False
        self._mode = ControlMode.SERVO_POSITION
        self._authority: AuthorityMode = "hold"
        self._epoch = 0
        self._observed = _TickObservation()
        self._accepted: OrderedDict[int, AcceptedTarget] = OrderedDict()

    def connect(self) -> bool:
        with self._lock:
            self._client.connect()
            self._connected = True
        return True

    def disconnect(self) -> None:
        with self._lock:
            self._enabled = False
            self._connected = False
            self._client.close()

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def activate(self) -> bool:
        return self.write_enable(True)

    def deactivate(self) -> bool:
        return self.write_enable(False)

    def transition(self, mode: AuthorityMode) -> int:
        with self._lock:
            self._epoch += 1
            self._authority = mode
            return self._epoch

    def snapshot(self) -> tuple[StateFrame, AcceptedTarget | None]:
        with self._lock:
            frame = self._client.read()
            target = self._accepted.get(frame.metadata.applied_action_sequence)
            if target is not None and target.generation != frame.metadata.episode_id:
                target = None
            return frame, target

    def get_info(self) -> ManipulatorInfo:
        return ManipulatorInfo("I2RT", "YAM sim2 artifact", len(self._joints))

    def get_dof(self) -> int:
        return len(self._joints)

    def get_limits(self) -> JointLimits:
        return JointLimits(
            position_lower=[joint.lower for joint in self._joints],
            position_upper=[joint.upper for joint in self._joints],
            velocity_max=[joint.max_velocity for joint in self._joints],
        )

    def set_control_mode(self, mode: ControlMode) -> bool:
        if mode not in (ControlMode.POSITION, ControlMode.SERVO_POSITION):
            return False
        with self._lock:
            self._mode = mode
        return True

    def get_control_mode(self) -> ControlMode:
        with self._lock:
            return self._mode

    def read_joint_positions(self) -> list[float]:
        with self._lock:
            self._observed.frame = self._client.read()
            self._observed.epoch = self._epoch
            return [float(value) for value in self._observed.frame.values["position"]]

    def read_joint_velocities(self) -> list[float]:
        with self._lock:
            if self._observed.frame is None:
                raise RuntimeError("read joint positions before velocity")
            return [float(value) for value in self._observed.frame.values["velocity"]]

    def observed_generation(self) -> int:
        """Reset generation of this control thread's joint-state snapshot."""
        if self._observed.frame is None:
            raise RuntimeError("read joint positions before their reset generation")
        return self._observed.frame.metadata.episode_id

    def read_joint_efforts(self) -> list[float]:
        with self._lock:
            if self._observed.frame is None:
                raise RuntimeError("read joint positions before effort")
            return [float(value) for value in self._observed.frame.values["effort"]]

    def write_joint_positions(self, positions: list[float], velocity: float = 1.0) -> bool:
        with self._lock:
            frame = self._observed.frame
            if not self._enabled or frame is None or self._observed.epoch != self._epoch:
                return False
            values = np.asarray(positions, dtype=np.float64)
            if values.shape != (len(PUBLIC_JOINTS),) or not np.isfinite(values).all():
                return False
            if any(not j.lower <= v <= j.upper for j, v in zip(self._joints, values, strict=True)):
                return False
            timestamp = time.time()
            sequence = self._client.write_positions(positions, observed=frame, source_ts=timestamp)
            self._accepted[sequence] = AcceptedTarget(
                sequence,
                timestamp,
                tuple(positions),
                self._authority,
                self._epoch,
                frame.metadata.episode_id,
            )
            while len(self._accepted) > 512:
                self._accepted.popitem(last=False)
            return True

    def write_joint_velocities(self, velocities: list[float]) -> bool:
        return False

    def write_stop(self) -> bool:
        self.transition("hold")
        return True

    def write_enable(self, enable: bool) -> bool:
        with self._lock:
            self._enabled = enable
            self._epoch += 1
        return True

    def read_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def read_state(self) -> dict[str, int]:
        return {"enabled": int(self.read_enabled())}

    def read_error(self) -> tuple[int, str]:
        return 0, ""

    def write_clear_errors(self) -> bool:
        return True

    def read_cartesian_position(self) -> dict[str, float] | None:
        return None

    def write_cartesian_position(self, pose: dict[str, float], velocity: float = 1.0) -> bool:
        return False

    def read_force_torque(self) -> list[float] | None:
        return None
