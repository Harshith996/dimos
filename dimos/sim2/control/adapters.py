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

"""ControlCoordinator whole-body adapter backed by the sim2 robot client.

The whole-body registry creates this adapter inside the coordinator process
under the name ``sim2``. It attaches to a robot in the blueprint's
``SimulationModule`` and speaks the same ``WholeBodyAdapter`` protocol as the
hardware adapters, so the coordinator and its tasks do not change between
hardware and simulation. Motor commands and feedback travel over DimOS Zenoh
through ``SimRobotClient``; nothing here owns physics or shared memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time

import numpy as np

from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.spec import POS_STOP, VEL_STOP, IMUState, MotorCommand, MotorState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.protocol.rpc.rpc_utils import RemoteError
from dimos.sim2.client import SimRobotClient, connect_with_retry
from dimos.sim2.messages import RobotFeedback
from dimos.sim2.world.spec import SimulationSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class Sim2WholeBodyAdapter:
    """Joint-level IO for one simulated robot, sampled coherently per tick.

    ``read_motor_states`` takes one sample from the world and ``read_imu``
    returns the IMU of that same sample, so a policy observes joints and IMU
    from one physics tick. ``write_motor_commands`` names that sample as the
    basis of the command, which lets the world reject commands that predate a
    reset.

    Args:
        simulation: The blueprint's ``SimulationModule``, injected by the
            coordinator. Required.
        hardware_id: The coordinator's hardware id. It is the robot id in the
            world unless ``address`` names another.
        address: The robot id in the world when it differs from ``hardware_id``.
        dof: Expected joint count, checked against the world's description.
        joint_names: Expected joint order, checked against the world's description.
        connect_timeout: Seconds to wait for the world to describe the robot.
        feedback_timeout: Seconds after which a sample counts as stale.
        command_timeout: Seconds a command stays valid in the world.
    """

    def __init__(
        self,
        *,
        simulation: SimulationSpec | None = None,
        hardware_id: str = "",
        address: str | Path | None = None,
        dof: int | None = None,
        joint_names: Sequence[str] | None = None,
        connect_timeout: float = 30.0,
        feedback_timeout: float = 0.25,
        command_timeout: float = 0.25,
        **_: object,
    ) -> None:
        if simulation is None:
            raise ValueError("the sim2 adapter requires a SimulationModule in the blueprint")
        robot_id = str(address) if address else hardware_id
        if not robot_id:
            raise ValueError("the sim2 adapter requires a robot id (hardware_id or address)")
        if connect_timeout < 0:
            raise ValueError("connect_timeout must be nonnegative")
        self._robot_id = robot_id
        self._dof = dof
        self._connect_timeout = connect_timeout
        self._client = SimRobotClient(
            world=simulation,
            robot_id=robot_id,
            joint_names=joint_names,
            writable=True,
            feedback_timeout=feedback_timeout,
            command_timeout=command_timeout,
        )
        self._sample: RobotFeedback | None = None
        self._connected = False
        self._feedback_lost = False

    def connect(self) -> bool:
        """Attach to the robot, waiting for the world to publish its endpoint."""
        try:
            connect_with_retry(self._client, self._connect_timeout)
        except (RuntimeError, TimeoutError, OSError, RemoteError) as error:
            logger.error("sim2 adapter could not attach", robot=self._robot_id, error=str(error))
            return False
        names = self._client.joint_names
        if self._dof is not None and len(names) != self._dof:
            self._client.close()
            raise ValueError(
                f"sim2 robot {self._robot_id!r} has {len(names)} joints, expected {self._dof}"
            )
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._sample = None
        self._client.close()

    def is_connected(self) -> bool:
        return self._connected

    def has_motor_states(self) -> bool:
        return self._read() is not None

    def read_motor_states(self) -> list[MotorState]:
        sample = self._read()
        if sample is None:
            return [MotorState()] * len(self._client.joint_names) if self._connected else []
        joints = sample.joints
        return [
            MotorState(q=q, dq=dq, tau=tau)
            for q, dq, tau in zip(joints.position, joints.velocity, joints.effort, strict=True)
        ]

    def read_imu(self) -> IMUState:
        """The IMU of the last joint sample; rpy stays zero like the hardware adapter."""
        sample = self._sample
        if sample is None or sample.imu is None:
            return IMUState()
        orientation = sample.imu.orientation
        return IMUState(
            quaternion=(orientation.w, orientation.x, orientation.y, orientation.z),
            gyroscope=sample.imu.angular_velocity.to_tuple(),
            accelerometer=sample.imu.linear_acceleration.to_tuple(),
        )

    def get_limits(self) -> JointLimits | None:
        return None

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        """Forward one complete PD/feedforward command based on the last sample."""
        sample = self._sample
        if sample is None:
            return False
        if len(commands) != len(sample.joints.position):
            raise ValueError(
                f"expected {len(sample.joints.position)} motor commands, got {len(commands)}"
            )
        values = np.array([[c.q, c.dq, c.kp, c.kd, c.tau] for c in commands], dtype=np.float64)
        q, dq, kp, kd, tau = values.T.copy()
        # Unitree "no command" sentinels: a joint without a target gets no gain on it.
        kp[q == POS_STOP] = 0.0
        q[q == POS_STOP] = 0.0
        kd[dq == VEL_STOP] = 0.0
        dq[dq == VEL_STOP] = 0.0
        command = MotorCommandArray(
            q=q.tolist(),
            dq=dq.tolist(),
            kp=kp.tolist(),
            kd=kd.tolist(),
            tau=tau.tolist(),
            timestamp=time.time(),
        )
        try:
            self._client.write_motors(command, observed=sample)
        except (RuntimeError, TimeoutError, ValueError) as error:
            logger.warning("sim2 command rejected", robot=self._robot_id, error=str(error))
            return False
        return True

    def _read(self) -> RobotFeedback | None:
        if not self._connected:
            return None
        try:
            self._sample = self._client.read()
        except (RuntimeError, TimeoutError) as error:
            self._sample = None
            if not self._feedback_lost:
                logger.warning("sim2 feedback unavailable", robot=self._robot_id, error=str(error))
            self._feedback_lost = True
            return None
        if self._feedback_lost:
            logger.info("sim2 feedback restored", robot=self._robot_id)
            self._feedback_lost = False
        return self._sample
