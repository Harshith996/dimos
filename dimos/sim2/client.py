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

"""Internal motor IO for a robot connection or policy host, not a DimOS Module."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
import math
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.std_msgs.String import String
from dimos.protocol.rpc.rpc_utils import RemoteError
from dimos.sim2.messages import FrameMetadata, RobotFeedback
from dimos.sim2.streams import SimStream
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.sim2.world.spec import SimulationSpec

logger = setup_logger()


def connect_with_retry(client: SimRobotClient, timeout: float, *, poll: float = 0.2) -> None:
    """Connect once the world describes the robot and publishes a fresh sample.

    Modules start in parallel, so a consumer can ask before the world has
    described its robots or before physics has published its first live sample.
    Every failure before ``timeout`` elapses is retried; the last one is raised.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.connect(timeout=max(0.0, deadline - time.monotonic()))
            return
        except (RuntimeError, TimeoutError, OSError, RemoteError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(poll)


class SimRobotClient:
    """Motor device attached by world and robot identity.

    One writable client owns the complete actuator set. Commands name the observation
    they were computed from, so reset invalidates in-flight policy results as well as
    held targets. Feedback uses DimOS messages, including XYZW quaternions.
    ``start`` publishes callbacks; ``connect`` lets a local policy read samples itself.
    """

    def __init__(
        self,
        *,
        world: SimulationSpec,
        robot_id: str,
        joint_names: Sequence[str] | None = None,
        frame_id: str | None = None,
        publish_joints: Callable[[JointState], None] | None = None,
        publish_imu: Callable[[Imu], None] | None = None,
        publish_odom: Callable[[PoseStamped], None] | None = None,
        publish_fault: Callable[[String], None] | None = None,
        rate_hz: float = 100.0,
        damping: float | None = None,
        writable: bool = True,
        feedback_timeout: float = 0.25,
        command_timeout: float = 0.25,
    ) -> None:
        if any(
            not math.isfinite(v) or v <= 0 for v in (feedback_timeout, command_timeout, rate_hz)
        ):
            raise ValueError("client timeouts must be finite and positive")
        if damping is not None and (not math.isfinite(damping) or damping < 0):
            raise ValueError("damping must be finite and nonnegative")
        self._world = world
        self._robot_id = robot_id
        self._joint_names = None if joint_names is None else tuple(joint_names)
        self._frame_id = frame_id
        self._publish_joints = publish_joints
        self._publish_imu = publish_imu
        self._publish_odom = publish_odom
        self._publish_fault = publish_fault
        self._rate_hz = rate_hz
        self._damping = damping
        self._estopped = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.feedback_timeout = feedback_timeout
        self.command_timeout = command_timeout
        self._channel: SimStream[RobotFeedback] | None = None
        self._last_observation: RobotFeedback | None = None
        self._lock = threading.RLock()
        self._writable = writable

    def connect(self, timeout: float = 30.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("connect timeout must be finite and nonnegative")
        with self._lock:
            if self._channel is not None:
                raise RuntimeError("sim2 client is already connected")
            descriptor = self._world.describe_robot(self._robot_id)
            if not descriptor.dof:
                raise ValueError("SimRobotClient requires a robot, not a world snapshot")
            if self._joint_names is not None and self._joint_names != descriptor.joint_names:
                raise ValueError("simulated motor joint order does not match the connection")
            try:
                self._channel = SimStream.attach(
                    descriptor, writable=self._writable, timeout=timeout
                )
                self.read()
            except BaseException:
                self.close()
                raise

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._require_channel().descriptor.joint_names

    def start(self) -> None:
        self.connect()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, name="sim2-feedback", daemon=True
        )
        self._thread.start()

    def _publish_loop(self) -> None:
        sequence = -1
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                sample = self._read()
                if sample.metadata.sequence != sequence:
                    sequence = sample.metadata.sequence
                    if self._publish_joints is not None:
                        joints = copy(sample.joints)
                        if self._frame_id is not None:
                            joints.frame_id = self._frame_id
                        self._publish_joints(joints)
                    if self._publish_imu is not None and sample.imu is not None:
                        imu = copy(sample.imu)
                        if self._frame_id is not None:
                            imu.frame_id = self._frame_id
                        self._publish_imu(imu)
                    if self._publish_odom is not None:
                        self._publish_odom(sample.odom)
                deadline += 1 / self._rate_hz
                now = time.monotonic()
                deadline = max(deadline, now)
                self._stop.wait(deadline - now)
        except Exception as error:
            logger.exception("Simulated motor feedback stopped")
            if self._damping is not None:
                try:
                    self.set_estop(True)
                except (RuntimeError, TimeoutError):
                    logger.exception("Simulator unavailable for damping stop")
            if self._publish_fault is not None:
                self._publish_fault(String(str(error)))

    def _require_channel(self) -> SimStream[RobotFeedback]:
        if self._channel is None:
            raise RuntimeError("sim2 client is disconnected")
        if self._channel.lifecycle != "ready":
            raise RuntimeError(f"sim2 robot is {self._channel.lifecycle}")
        return self._channel

    def _validate_observation(self, frame: RobotFeedback) -> None:
        age = time.monotonic() - frame.monotonic_time
        if not 0 <= age <= self.feedback_timeout:
            raise TimeoutError("sim2 robot feedback is stale")

    def read(self) -> RobotFeedback:
        """Read one coherent joint/IMU/base sample; never mix separate physics ticks."""
        with self._lock:
            self._last_observation = self._read()
            return self._last_observation

    def _read(self) -> RobotFeedback:
        with self._lock:
            channel = self._require_channel()
            frame = channel.read_observation()
            if frame is None:
                raise RuntimeError("sim2 robot has no observation")
            self._validate_observation(frame)
            return frame

    def write(self, command: MotorCommandArray) -> None:
        """Forward a timestamped motor command without changing its acquisition time."""
        self.write_motors(command, observed=self.read())

    def write_motors(self, command: MotorCommandArray, *, observed: RobotFeedback) -> int:
        """Write complete PD/feedforward targets in descriptor joint order."""
        values = np.asarray([command.q, command.dq, command.kp, command.kd, command.tau])
        descriptor = self._require_channel().descriptor
        if values.shape != (5, descriptor.dof) or not np.isfinite(values).all():
            raise ValueError(f"expected {descriptor.dof} finite q/dq/kp/kd/tau values")
        if np.any(values[2:4] < 0):
            raise ValueError("motor gains must be nonnegative")
        position_servos = np.array([mode == "position" for mode in descriptor.actuator_modes])
        if np.any(values[1:, position_servos] != 0):
            raise ValueError(
                "native position servos accept positions, not velocity/gain/effort overrides"
            )
        q, dq, kp, kd, effort = values
        return self._write(
            dict(position=q, velocity=dq, kp=kp, kd=kd, effort=effort),
            observed,
            enabled=True,
            source_ts=command.timestamp,
        )

    def write_positions(
        self, positions: Sequence[float], *, observed: RobotFeedback, source_ts: float | None = None
    ) -> int:
        """Write native joint targets, including grippers in their declared units."""
        q = np.asarray(positions, dtype=np.float64)
        descriptor = self._require_channel().descriptor
        if q.shape != (descriptor.dof,) or not np.isfinite(q).all():
            raise ValueError(f"expected {descriptor.dof} finite joint positions")
        if any(
            mode == "effort" and gains[0] == 0
            for mode, gains in zip(
                descriptor.actuator_modes, descriptor.position_gains, strict=True
            )
        ):
            raise ValueError("position requests require a configured servo; send PD motor commands")
        kp, kd = np.asarray(descriptor.position_gains).T
        return self._write(
            dict(position=q, velocity=np.zeros_like(q), kp=kp, kd=kd, effort=np.zeros_like(q)),
            observed,
            enabled=True,
            source_ts=source_ts,
        )

    def disable(self, *, observed: RobotFeedback) -> int:
        """Disable effort outputs; native position servos hold their current position.

        This is device control, not a promise that a floating robot remains upright.
        """
        dof = len(self.joint_names)
        values = dict(
            position=observed.joints.position,
            velocity=np.zeros(dof),
            effort=np.zeros(dof),
        )
        values.update(kp=np.zeros(dof), kd=np.zeros(dof))
        return self._write(values, observed, enabled=False)

    def set_estop(self, estopped: bool) -> None:
        """Latch motor damping until world restart, even if feedback is stale."""
        with self._lock:
            if not estopped:
                if self._estopped or self.read().estopped:
                    raise RuntimeError("sim2 emergency stop is latched; restart to recover")
                return
            if self._damping is None:
                raise ValueError("this motor device has no configured damping stop")
            channel = self._require_channel()
            self._estopped = True
            channel.halt([self._damping] * channel.descriptor.dof)

    def _write(
        self,
        values: dict[str, Any],
        observed: RobotFeedback,
        *,
        enabled: bool,
        source_ts: float | None = None,
    ) -> int:
        with self._lock:
            channel = self._require_channel()
            if not self._writable:
                raise PermissionError("sim2 client is read-only")
            if self._estopped or observed.estopped:
                raise RuntimeError("sim2 emergency stop is latched; restart to recover")
            if observed is not self._last_observation:
                raise ValueError("command must use this client's latest observation")
            if observed.metadata.episode_id != channel.episode_id:
                raise RuntimeError("sim2 robot was reset; read a new observation before commanding")
            self._validate_observation(observed)
            issued = time.time() if source_ts is None else source_ts
            if not math.isfinite(issued) or issued < observed.reset_at:
                raise RuntimeError("sim2 command predates reset")
            remaining = self.command_timeout - max(0.0, time.time() - issued)
            if remaining <= 0:
                raise TimeoutError("sim2 command is stale")
            values.update(enabled=[int(enabled)], expires_at=[time.monotonic() + remaining])
            return channel.publish_action(
                values,
                FrameMetadata(
                    0,
                    observed.metadata.episode_id,
                    observed.metadata.physics_tick,
                    0,
                    observed.metadata.sim_time,
                ),
                source_ts=issued,
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            if self._thread.is_alive():
                raise RuntimeError("sim2 feedback publisher did not stop")
        self._thread = None
        with self._lock:
            if self._channel is not None:
                self._channel.close()
                self._channel = None
                self._last_observation = None

    def stop(self) -> None:
        self.close()

    def __enter__(self) -> SimRobotClient:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
